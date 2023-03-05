# install some dependency
# pip3 install openvino-dev openvino-dev[onnx]

# prepare
# mkdir -p save_dir/hf
# mkdir -p save_dir/onnx
# mkdir -p save_dir/ov

# # train
# python3 train.py

# # convert
# python3 onnx_convert.py
# sh openvino_convert.sh

# predict
python3 torch_predict.py
python3 onnx_predict.py
python3 openvino_predict.py
