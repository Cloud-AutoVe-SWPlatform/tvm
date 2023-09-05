import os
from copy import deepcopy

import numpy as np
import onnx
from absl import app, flags
from PIL import Image

import tvm
from tvm import auto_scheduler, relay
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "model", "tail_det", ["tail_det", "taillight_unet", "taillight_3Dconv"], "Choose model."
)


def tvm_amp(mod, params):
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

    mod = relay.transform.EliminateCommonSubexpr()(mod)
    mod = relay.transform.FoldConstant()(mod)

    return mod


def main(_):
    os.environ["PATH"] += os.pathsep + "/usr/local/cuda/bin/"

    # Prepare model
    model_path = f"{FLAGS.model}.onnx"
    onnx_model = onnx.load(model_path)
    input_node = onnx_model.graph.input[0]
    input_name = input_node.name
    input_shape = tuple(getattr(d, "dim_value", 0) for d in input_node.type.tensor_type.shape.dim)
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    mod = relay.transform.DynamicToStatic()(mod)
    mod = tvm_amp(mod, params)
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
    tasks, task_weights = auto_scheduler.extract_tasks(deepcopy(mod["main"]), params, target)

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
    rng = np.random.default_rng(42)
    data_tvm = tvm.nd.array(rng.uniform(size=input_shape).astype(dtype))
    module.set_input(input_name, data_tvm)

    # Evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", dev, repeat=50)
    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    print(f"Mean inference time (std dev): {np.mean(prof_res):.2f} ms ({np.std(prof_res):.2f} ms)")


if __name__ == "__main__":
    app.run(main)
