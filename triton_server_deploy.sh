# $1 indicates the seq_len

# prepare related files

# copy model
rm -rf save_dir/triton_models/bert_openvino
mkdir -p save_dir/triton_models/bert_openvino/1
cp save_dir/ov/$1/* save_dir/triton_models/bert_openvino/1

# copy model configuration file
cp triton_models/bert_openvino/config.pbtxt save_dir/triton_models/bert_openvino
sed -i 's/128/'"$1"'/g' ./save_dir/triton_models/bert_openvino/config.pbtxt

# start triton server docker
docker run --runtime=nvidia\
           --rm\
           --detach\
           --name tritonserver_openvino_backend\
           -p 8000:8000\
           -p 8001:8001\
           -p 8002:8002\
           -v $(pwd):$(pwd)\
           -w $(pwd)\
           --shm-size="4g"\
           nvcr.io/nvidia/tritonserver:22.09-py3 tritonserver --model-store $(pwd)/save_dir/triton_models
