import os
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torchvision
from absl import app, flags
from PIL import Image
from torchvision import transforms

import tvm
from tvm import auto_scheduler, relay
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata

FLAGS = flags.FLAGS

flags.DEFINE_bool("cached", True, "Use cached model.")
flags.DEFINE_enum("backend", "opencv_cpu", ["opencv_cpu", "tvm_gpu"], "Choose backend.")
flags.DEFINE_enum(
    "model", "mobilenet_v2", ["mobilenet_v2", "resnet50", "efficientnet_v2_s"], "Choose model."
)


def tvm_amp(mod, params, to_nhwc=False):
    mod = relay.transform.EliminateCommonSubexpr()(mod)

    BindPass = relay.transform.function_pass(
        lambda fn, new_mod, ctx: relay.build_module.bind_params_by_name(fn, params),
        opt_level=1,
    )
    mod = BindPass(mod)
    mod = relay.transform.FoldConstant()(mod)

    mod = relay.transform.CombineParallelBatchMatmul()(mod)
    mod = relay.transform.FoldConstant()(mod)

    mod = relay.transform.InferType()(mod)
    mod = relay.transform.ToMixedPrecision()(mod)

    if to_nhwc:
        desired_layouts = {
            k: ["NHWC", "default"] for k in ["nn.conv2d", "nn.max_pool2d", "qnn.conv2d"]
        }
        mod = relay.transform.ConvertLayout(desired_layouts)(mod)

    mod = relay.transform.EliminateCommonSubexpr()(mod)
    mod = relay.transform.FoldConstant()(mod)

    return mod


def main(_):
    os.environ["PATH"] += os.pathsep + "/usr/local/cuda/bin/"

    # Prepare test data
    img_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    img_path = download_testdata(img_url, "dog.jpg", module="data")
    img = Image.open(img_path)
    preprocess_input = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    data = preprocess_input(img)
    data = np.expand_dims(data, 0)

    synset_url = (
        "https://gist.githubusercontent.com/zhreshold/"
        "4d0b62f3d01426887599d4f7ede23ee5/raw/"
        "596b27d23537e5a1b5751d2b0481ef172f58b539/"
        "imagenet1000_clsid_to_human.txt"
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        synset = eval(f.read())

    dummy_input = torch.randn(data.shape)
    input_name = "input_1"
    repeat = 50

    if FLAGS.backend == "opencv_cpu":
        # Prepare model
        onnx_file = Path(f"{FLAGS.model}.onnx")
        if not (FLAGS.cached and onnx_file.exists()):
            torch_model = (
                timm.create_model("tf_efficientnetv2_s", pretrained=True)
                if FLAGS.model == "efficientnet_v2_s"
                else getattr(torchvision.models, FLAGS.model)(weights="DEFAULT")
            )
            torch_model.eval()
            torch.onnx.export(
                torch_model, dummy_input, onnx_file, input_names=[input_name], opset_version=11
            )
        opencv_net = cv2.dnn.readNetFromONNX(str(onnx_file))
        opencv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        opencv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        opencv_net.setInput(data)

        # Evaluate
        print("Evaluate inference time cost...")
        opencv_prof_res = []
        opencv_out = opencv_net.forward()
        for _ in range(repeat):
            opencv_net.forward()
            opencv_prof_res.append(opencv_net.getPerfProfile()[0])
        opencv_prof_res = np.array(opencv_prof_res) * 1000.0 / cv2.getTickFrequency()
        print(
            f"OpenCV Mean inference time (std dev): {np.mean(opencv_prof_res):.2f} ms ({np.std(opencv_prof_res):.2f} ms)"
        )
        top1_opencv = np.argmax(opencv_out)
        print(f"OpenCV top-1 id: {top1_opencv}, class name: {synset[top1_opencv]}")
    else:
        # Prepare model
        lib_file = Path(f"{FLAGS.model}.tar")
        target = tvm.target.Target("cuda -arch=sm_72", host="llvm -mcpu=carmel")
        if not (FLAGS.cached and lib_file.exists()):
            torch_model = (
                timm.create_model("tf_efficientnetv2_s", pretrained=True)
                if FLAGS.model == "efficientnet_v2_s"
                else getattr(torchvision.models, FLAGS.model)(weights="DEFAULT")
            )
            torch_model.eval()

            scripted_torch_model = torch.jit.trace(torch_model, dummy_input).eval()
            shape_list = [(input_name, data.shape)]
            mod, params = relay.frontend.from_pytorch(scripted_torch_model, shape_list)
            mod = tvm_amp(mod, params, to_nhwc=True)
            params = None

            # Compile with the history best
            log_file = f"{FLAGS.model}.json"
            print("Compile...")
            with auto_scheduler.ApplyHistoryBest(log_file):
                with tvm.transform.PassContext(
                    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
                ):
                    lib = relay.build(mod, target=target)
            lib.export_library(lib_file)
        lib = tvm.runtime.load_module(lib_file)

        # Create graph executor
        dev = tvm.device(str(target))
        module = graph_executor.GraphModule(lib["default"](dev))
        dtype = "float32"
        data_tvm = tvm.nd.array(data.astype(dtype))
        module.set_input(input_name, data_tvm)

        # Evaluate
        print("Evaluate inference time cost...")

        ftimer = module.module.time_evaluator("run", dev, repeat=repeat)
        tvm_prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        print(
            f"Relay Mean inference time (std dev): {np.mean(tvm_prof_res):.2f} ms ({np.std(tvm_prof_res):.2f} ms)"
        )
        module.run()
        tvm_out = module.get_output(0)
        top1_tvm = np.argmax(tvm_out.asnumpy())
        print(f"Relay top-1 id: {top1_tvm}, class name: {synset[top1_tvm]}")


if __name__ == "__main__":
    app.run(main)
