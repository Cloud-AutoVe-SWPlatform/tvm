import os
from copy import deepcopy

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

    # Prepare model
    torch_model = (
        timm.create_model("tf_efficientnetv2_s", pretrained=True)
        if FLAGS.model == "efficientnet_v2_s"
        else getattr(torchvision.models, FLAGS.model)(weights="DEFAULT")
    )
    torch_model.eval()
    scripted_torch_model = torch.jit.trace(torch_model, torch.randn(data.shape)).eval()
    input_name = "input_1"
    shape_list = [(input_name, data.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_torch_model, shape_list)
    mod = tvm_amp(mod, params, to_nhwc=True)
    params = None

    # Extract tasks from the network
    target = tvm.target.Target(
        "cuda -arch=sm_72", host="llvm -mtriple=aarch64-linux-gnu -mcpu=carmel"
    )
    device_key = "xavier"
    host = "0.0.0.0"
    port = 9190
    log_file = f"{FLAGS.model}.json"

    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(deepcopy(mod), params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    def run_tuning():
        print("Begin tuning...")
        remote_runner = auto_scheduler.RPCRunner(
            key=device_key, host=host, port=port, repeat=1, min_repeat_ms=300, timeout=600
        )

        tuner = auto_scheduler.TaskScheduler(
            tasks, task_weights, strategy="round-robin", load_log_file=log_file
        )
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=len(tasks) * 2000,
            builder=auto_scheduler.LocalBuilder(timeout=60),
            runner=remote_runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option, per_task_early_stopping=600)

    run_tuning()

    # Compile with the history best
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(mod, target=target)
    lib_file = f"{FLAGS.model}.tar"
    lib.export_library(lib_file)

    remote = tvm.auto_scheduler.utils.request_remote(
        device_key=device_key, host=host, port=port, timeout=180
    )
    dev = remote.device(str(target))

    remote.upload(lib_file)
    lib = remote.load_module(lib_file)

    # Create graph executor
    module = graph_executor.GraphModule(lib["default"](dev))
    dtype = "float32"
    data_tvm = tvm.nd.array(data.astype(dtype))
    module.set_input(input_name, data_tvm)

    # Evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", dev, repeat=50)
    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    print(f"Mean inference time (std dev): {np.mean(prof_res):.2f} ms ({np.std(prof_res):.2f} ms)")

    module.run()
    tvm_out = module.get_output(0)
    top1_tvm = np.argmax(tvm_out.asnumpy())

    print(f"Relay top-1 id: {top1_tvm}, class name: {synset[top1_tvm]}")
    # confirm correctness with torch output
    with torch.no_grad():
        torch_img = torch.from_numpy(data)
        output = torch_model(torch_img)

        # Get top-1 result for PyTorch
        top1_torch = np.argmax(output.numpy())

    print(f"Torch top-1 id: {top1_torch}, class name: {synset[top1_torch]}")


if __name__ == "__main__":
    app.run(main)
