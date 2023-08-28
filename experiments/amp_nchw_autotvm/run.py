from copy import deepcopy

from PIL import Image
import numpy as np
import timm
import torch
import torchvision
from torchvision import transforms
import tvm
from tvm import relay, auto_scheduler, autotvm
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor
from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner


def tvm_amp(mod, params, to_nhwc=False):
    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)

    BindPass = tvm.relay.transform.function_pass(
        lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(fn, params),
        opt_level=1,
    )
    mod = BindPass(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)

    mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)

    mod = tvm.relay.transform.InferType()(mod)
    mod = tvm.relay.transform.ToMixedPrecision()(mod)

    if to_nhwc:
        desired_layouts = {"nn.conv2d": ["NHWC", "default"], "qnn.conv2d": ["NHWC", "default"]}
        mod = relay.transform.ConvertLayout(desired_layouts)(mod)

    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)

    return mod


if __name__ == "__main__":
    import os

    from absl import app, flags

    os.environ["PATH"] += os.pathsep + "/usr/local/cuda/bin/"

    FLAGS = flags.FLAGS

    flags.DEFINE_enum(
        "model", "mobilenet_v2", ["mobilenet_v2", "resnet50", "efficientnet_v2_s"], "Choose model."
    )

    def main(_):
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

        torch_model = (
            timm.create_model("tf_efficientnetv2_s", pretrained=True)
            if FLAGS.model == "efficientnet_v2_s"
            else getattr(torchvision.models, FLAGS.model)(weights="DEFAULT")
        )
        torch_model.eval()
        scripted_torch_model = torch.jit.trace(torch_model, torch.randn(data.shape)).eval()
        shape_list = [("input_1", data.shape)]
        mod, params = relay.frontend.from_pytorch(scripted_torch_model, shape_list)
        mod = tvm_amp(mod, params, to_nhwc=False)
        params = None

        target = tvm.target.Target(
            "cuda -arch=sm_72", host="llvm -mtriple=aarch64-linux-gnu -mcpu=carmel"
        )
        device_key = "xavier"
        host = "0.0.0.0"
        port = 9190

        log_file = f"{FLAGS.model}.json"
        lib_file = f"{FLAGS.model}.tar"

        tuning_option = {
            "log_filename": log_file,
            "tuner": "xgb",
            "n_trial": 4000,
            "early_stopping": 600,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=60),
                runner=autotvm.RPCRunner(
                    device_key,
                    host=host,
                    port=port,
                    repeat=1,
                    min_repeat_ms=300,
                    timeout=600
                ),
            ),
        }

        print("Extract tasks...")
        tasks = autotvm.task.extract_from_program(
            mod["main"],
            target=target,
            params=params,
        )

        def tune_tasks(
            tasks,
            measure_option,
            tuner="xgb",
            n_trial=1000,
            early_stopping=None,
            log_filename="tuning.log",
            use_transfer_learning=True,
        ):
            for i, tsk in enumerate(reversed(tasks)):
                prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

                # create tuner
                if tuner == "xgb" or tuner == "xgb-rank":
                    tuner_obj = XGBTuner(tsk, loss_type="rank")
                elif tuner == "ga":
                    tuner_obj = GATuner(tsk, pop_size=50)
                elif tuner == "random":
                    tuner_obj = RandomTuner(tsk)
                elif tuner == "gridsearch":
                    tuner_obj = GridSearchTuner(tsk)
                else:
                    raise ValueError("Invalid tuner: " + tuner)

                if use_transfer_learning:
                    if os.path.isfile(log_file):
                        tuner_obj.load_history(autotvm.record.load_from_file(log_file))

                # do tuning
                tsk_trial = min(n_trial, len(tsk.config_space))
                tuner_obj.tune(
                    n_trial=tsk_trial,
                    early_stopping=early_stopping,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                        autotvm.callback.log_to_file(log_file),
                    ],
                )

        print("Tuning...")
        tune_tasks(tasks, **tuning_option)

        # Compile with the history best
        print("Compile...")
        with autotvm.apply_history_best(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target)
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
        module.set_input("input_1", data_tvm)

        # Evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, repeat=50)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

        module.run()
        tvm_out = module.get_output(0)
        top1_tvm = np.argmax(tvm_out.asnumpy())

        synset_url = "".join(
            [
                "https://gist.githubusercontent.com/zhreshold/",
                "4d0b62f3d01426887599d4f7ede23ee5/raw/",
                "596b27d23537e5a1b5751d2b0481ef172f58b539/",
                "imagenet1000_clsid_to_human.txt",
            ]
        )
        synset_name = "imagenet1000_clsid_to_human.txt"
        synset_path = download_testdata(synset_url, synset_name, module="data")
        with open(synset_path) as f:
            synset = eval(f.read())
        print("Relay top-1 id: {}, class name: {}".format(top1_tvm, synset[top1_tvm]))
        # confirm correctness with torch output
        with torch.no_grad():
            torch_img = torch.from_numpy(data)
            output = torch_model(torch_img)

            # Get top-1 result for PyTorch
            top1_torch = np.argmax(output.numpy())

        print("Torch top-1 id: {}, class name: {}".format(top1_torch, synset[top1_torch]))

    app.run(main)
