import subprocess
import sys
import os

def install(package, file_link=None):
    if file_link:
        output = subprocess.run([sys.executable, "-m", "pip", "install", package, "-f", file_link], capture_output=True)
    else:
        output = subprocess.run([sys.executable, "-m", "pip", "install", package], capture_output=True)
    
    return output

def get_cuda_version():
        
    output = subprocess.run("nvcc --version", shell=True, capture_output=True)
    print(output)
    
    if output.returncode > 0:
        hardware="cpu"
        
    else:
        parsed_output = output.stdout.decode("utf-8").split()[-1].split(".")
        major_version = parsed_output[0][1:]
        minor_version = parsed_output[1]
        hardware = "cu{}".format(major_version+minor_version)
    
    print("Using:", hardware)
        
    return hardware

def main():
    
    hardware = get_cuda_version()
    os.environ["CUDA"] = hardware
    # Install requirements
#     output = install("-r requirements.txt", file_link = "https://download.pytorch.org/whl/torch_stable.html https://pytorch-geometric.com/whl/torch-1.8.0+{}.html".format(hardware))
    package_1 = "torch_requirements.txt"
    package_2 = "geometric_requirements.txt"
    file_link_1 = "https://download.pytorch.org/whl/torch_stable.html"
    file_link_2 = "https://pytorch-geometric.com/whl/torch-1.8.0+{}.html".format(hardware)
    output = subprocess.run([sys.executable, "-m", "pip", "install", "-r", package_1, "-f", file_link_1], capture_output=True)
    print(output)
    
    output = subprocess.run([sys.executable, "-m", "pip", "install", "-r", package_2, "-f", file_link_2], capture_output=True)
    print(output)
    
    # Install faiss
    # Install pytorch3d
    
    # Install setup
    

#     torch_version = "torch==1.8.0+"+hardware
#     output = install(torch_version, "https://download.pytorch.org/whl/torch_stable.html")
#     print(output)
    
#     try:
#         import torch
#         print("Imported!")
#     except ModuleNotFoundError as err:
#         print(err)    
    
        
if __name__=="__main__":
    main()