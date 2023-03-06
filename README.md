# bert infer

```
# pytorch
torch infer first time - Elapsed time: 67.9009 ms
torch infer 100 times - Elapsed time: 5852.7300 ms

# onnxruntime
onnx infer first time - Elapsed time: 57.4057 ms
onnx infer 100 times - Elapsed time: 5424.8796 ms

# openvino + python frontend
ov infer first time - Elapsed time: 62.4363 ms
ov infer 100 times - Elapsed time: 4584.5890 ms

# untuned
tvm infer first time - Elapsed time: 277.3764 ms
tvm infer 100 times - Elapsed time: 14279.5329 ms

# autotvm tuned
tvm infer first time - Elapsed time: 83.5037 ms
tvm infer 100 times - Elapsed time: 4824.3027 ms

# autoscheduler tuned
tvm infer first time - Elapsed time: 194.1788 ms
tvm infer 100 times - Elapsed time: 14470.1133 ms

# triton server infer with openvino backend
triton infer first time - Elapsed time: 225.9994 ms
triton infer 100 times - Elapsed time: 2751.7860 ms
```



# tvm

```bash
docker run -it\
           --runtime=nvidia\
           --network=host\
           -v $(pwd):$(pwd)\
           -w $(pwd)\
           --privileged\
           --cap-add sys_ptrace\
           --security-opt seccomp=unconfined\
           --shm-size="16g"\
           nvcr.io/nvidia/tritonserver:22.09-pyt-python-py3

# install llvm-15
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

# install onednn
# download from here https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#onednn
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19137/l_onednn_p_2023.0.0.25399_offline.sh

# install onednn previous version via pip that compatible with tvm
pip3 install onednn-devel-cpu-tbb==2022.2.0

# build tvm
git clone --recursive https://github.com/apache/tvm tvm

# edit cmake/config.cmake

# turn llvm on
# set(USE_LLVM llvm-config-15)

# turn dnnl on and add include directory
# set(USE_DNNL /usr/local/)
# include_directories(/usr/local/include/)

mkdir build
cd build
cp ../cmake/config.cmake  .
pip3 install cmake
cmake ..
make -j48

# export TVM Path to PYTHONPATH
export PYTHONPATH=$(pwd)/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/opt/intel/oneapi/compiler/2023.0.0/linux/compiler/lib/intel64_lin/
pip3 install decorator onnx psutil scipy attrs tornado cloudpickle
pip3 install xgboost==1.5.2  # the version must lower than specified version due to compatibility

# download tophub manunally
git clone https://github.com/tlc-pack/tophub
cp -r tophub/tophub/ ~/.t
```

