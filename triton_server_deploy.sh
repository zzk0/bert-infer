# prepare related files
mkdir -p triton_models/bert_openvino/1
cp save_dir/ov/* triton_models/bert_openvino/1

# start triton server docker
docker run -it\
           --runtime=nvidia\
           -p 8000:8000\
           -p 8001:8001\
           -p 8002:8002\
           -v $(pwd):$(pwd)\
           -w $(pwd)\
           --shm-size="4g"\
           nvcr.io/nvidia/tritonserver:22.09-py3 tritonserver --model-store $(pwd)/triton_models
