import subprocess
import sys
import os

def install(package, file_link=None, r=False, e=False):
    if file_link:
        output = subprocess.run([sys.executable, "-m", "pip", "install", package, "-f", file_link], capture_output=True)
    elif r:
        output = subprocess.run([sys.executable, "-m", "pip", "install", "-r", package], capture_output=True)
    elif e:
        output = subprocess.run([sys.executable, "-m", "pip", "install", "-e", package], capture_output=True)
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
    output = install("requirements.txt", r=True)
    print(output)
        
    # Install setup
    output = install(".", e=True)
    print(output)
    
    # Install FAISS
    if hardware is "cpu":
        output = install("faiss-cpu")
    else:   
        output = install("faiss-gpu")
        print(output)
        output = install("cupy-cuda{}".format(hardware[2:]))
        print(output)
    
    # Install pytorch3d
    
    

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