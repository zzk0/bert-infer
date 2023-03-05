# bert infer

```
torch infer first time - Elapsed time: 67.9009 ms
torch infer 100 times - Elapsed time: 5852.7300 ms

onnx infer first time - Elapsed time: 57.4057 ms
onnx infer 100 times - Elapsed time: 5424.8796 ms

ov infer first time - Elapsed time: 62.4363 ms
ov infer 100 times - Elapsed time: 4584.5890 ms
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
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"'

# install onednn

# install onednn previous version that compatible with tvm

# build tvm
```
