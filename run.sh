# install some dependency
# pip3 install openvino-dev openvino-dev[onnx]

# train
# mkdir -p save_dir/hf
# python3 train.py

# inference
log_file="save_dir/result.log"
touch $log_file
seq_lens="16 32 64 128 256 512"
for seq_len in $seq_lens
do
    # write log
    echo "============ max_len = $seq_len ============" >> $log_file

    # prepare
    mkdir -p save_dir/onnx/$seq_len
    mkdir -p save_dir/ov/$seq_len

    # torch predict
    python3 torch_predict.py --max_len $seq_len | grep infer >> $log_file

    # onnx predict
    python3 onnx_convert.py --max_len $seq_len
    python3 onnx_predict.py --max_len $seq_len  | grep infer >> $log_file

    # openvino python predict
    sh openvino_convert.sh $seq_len
    python3 openvino_predict.py --max_len $seq_len  | grep infer >> $log_file

    # openvino tritonserver predict
    sh triton_server_deploy.sh $seq_len
    sleep 10
    python3 triton_predict.py --max_len $seq_len  | grep infer >> $log_file
    docker stop tritonserver_openvino_backend
done
