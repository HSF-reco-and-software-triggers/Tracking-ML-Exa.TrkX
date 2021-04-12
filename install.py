import subprocess
import sys
import os


def install(package, file_link=None, r=False, e=False):
    if file_link:
        output = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "-f", file_link],
            capture_output=True,
        )
    elif r:
        output = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", package], capture_output=True
        )
    elif e:
        output = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", package], capture_output=True
        )
    else:
        output = subprocess.run(
            [sys.executable, "-m", "pip", "install", package], capture_output=True
        )

    return output


def get_cuda_version():

    output = subprocess.run("nvcc --version", shell=True, capture_output=True)
    print(output)

    if output.returncode > 0:
        hardware = "cpu"

    else:
        parsed_output = output.stdout.decode("utf-8").split()[-1].split(".")
        major_version = parsed_output[0][1:]
        minor_version = parsed_output[1]
        hardware = "cu{}".format(major_version + minor_version)

    print("Using:", hardware)

    return hardware


def main():

    # Get CUDA version
    hardware = get_cuda_version()
    os.environ["CUDA"] = hardware

    # Install Pytorch
    if hardware == "cu102":
        output = install("torch")
    else:
        file_link = "https://download.pytorch.org/whl/torch_stable.html"
        output = install("torch==1.8.0+{}".format(hardware), file_link=file_link)
    print(output.stdout.decode("utf-8"))

    # Install requirements
    output = install("requirements.txt", r=True)
    print(output.stdout.decode("utf-8"))

    # Install setup
    output = install(".", e=True)
    print(output.stdout.decode("utf-8"))

    # Install FAISS
    if hardware == "cpu":
        output = install("faiss-cpu")
        print(output.stdout.decode("utf-8"))
    else:
        output = install("faiss-gpu")
        print(output.stdout.decode("utf-8"))
        output = install("cupy-cuda{}".format(hardware[2:]))
        print(output.stdout.decode("utf-8"))

    # Install Pytorch3d
    if hardware == "cpu":
        output = install("git+https://github.com/facebookresearch/pytorch3d.git@stable")
    else:
        # If using cuda, pip install SHOULD work
        output = install(
            "pytorch3d",
            file_link="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py3{}_{}_pyt180/download.html".format(
                sys.version_info.minor, hardware
            ),
        )

    print(output.stdout.decode("utf-8"))


if __name__ == "__main__":
    main()
