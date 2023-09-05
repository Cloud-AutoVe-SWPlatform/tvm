import os
import pathlib
import shutil
import tarfile
import tempfile

import numpy as np
import torch
from absl import app
from PIL import Image
from torchvision import models, transforms

import tvm
from tvm import auto_scheduler, relay
from tvm.contrib.download import download_testdata
from tvm.relay.backend import Executor, Runtime


def create_header_file(name, section, tensor_name, tensor_data, output_path):
    """
    This function generates a header file containing the data from the numpy array provided.
    """
    file_path = pathlib.Path(f"{output_path}/" + name).resolve()
    # Create header file with npy_data as a C array
    raw_path = file_path.with_suffix(".h").resolve()
    with open(raw_path, "w") as header_file:
        header_file.write(
            "#include <stddef.h>\n"
            f"const size_t {tensor_name}_len = {tensor_data.size};\n"
            f'float {tensor_name}_storage[] __attribute__((section("{section}"), aligned(16))) = '
        )
        header_file.write("{")
        for i in np.ndindex(tensor_data.shape):
            header_file.write(f"{tensor_data[i]}, ")
        header_file.write("};\n\n")


def create_headers(preprocessed_data):
    """
    This function generates C header files for the input and output arrays required to run inferences
    """
    img_data = np.asarray(preprocessed_data).astype("float32")

    # Create input header file
    input_data = img_data.astype(np.float32)
    create_header_file("inputs", ".data.tvm", "input", input_data, "src")
    # Create output header file
    output_data = np.zeros([1001], np.float32)
    create_header_file(
        "outputs",
        ".data.tvm",
        "output",
        output_data,
        "src",
    )


def create_labels_header(labels_file, section, output_path):
    """
    This function generates a header file containing the ImageNet labels as an array of strings
    """
    labels_path = pathlib.Path(labels_file).resolve()
    file_path = pathlib.Path(f"{output_path}/labels.h").resolve()

    with open(labels_path) as f:
        labels = f.readlines()

    with open(file_path, "w") as header_file:
        header_file.write(
            f'const char* const labels[] __attribute__((section("{section}"), aligned(16))) = {{'
        )

        for _, label in enumerate(labels):
            label = label.rstrip().replace("'", "\\'")
            header_file.write(f'"{label}",')

        header_file.write("};\n")


def main(_):
    # Prepare test data
    img_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    img_path = download_testdata(img_url, "dog.jpg", module="data")
    img = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    preprocessed_data = preprocess(img)
    data = np.expand_dims(preprocessed_data, 0)

    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_path = download_testdata(labels_url, "imagenet_classes.txt", module="data")

    # Prepare model
    torch_mobilenetv2 = models.mobilenet_v2(weights="DEFAULT")
    torch_mobilenetv2.eval()
    scripted_torch_mobilenetv2 = torch.jit.trace(torch_mobilenetv2, torch.randn(data.shape)).eval()
    input_name = "input_1"
    shape_list = [(input_name, data.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_torch_mobilenetv2, shape_list)

    # Compile with the history best
    target = tvm.target.Target("c -mcpu=carmel")
    executor = Executor("aot", {"unpacked-api": True, "interface-api": "c"})
    runtime = Runtime("crt")
    log_file = "torch-mobilenetv2-carmel.json"
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"tir.disable_vectorize": True, "relay.backend.use_auto_scheduler": True},
        ):
            module = relay.build(
                mod, target=target, params=params, executor=executor, runtime=runtime
            )

    # Preview code
    c_source_module = module.get_lib().imported_modules[0]
    assert c_source_module.type_key == "c", "tutorial is broken"
    c_source_code = c_source_module.get_source()
    first_few_lines = c_source_code.split("\n")[:10]
    print(*first_few_lines, sep="\n")

    # Export code
    fd, model_library_format_tar_path = tempfile.mkstemp()
    os.close(fd)
    os.unlink(model_library_format_tar_path)
    tvm.micro.export_model_library_format(module, model_library_format_tar_path)
    with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
        for m in tar_f.getmembers():
            if "template" not in m.name:
                tar_f.extract(m, path="src/module")
    os.unlink(model_library_format_tar_path)

    create_headers(preprocessed_data)
    create_labels_header(labels_path, ".rodata.tvm", "src")
    shutil.copyfile("main.c.template", "src/main.c")
    shutil.copyfile("crt_config.h.template", "src/crt_config.h")


if __name__ == "__main__":
    app.run(main)
