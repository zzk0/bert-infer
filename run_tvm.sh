# install some dependency
# pip3 install openvino-dev openvino-dev[onnx]

# inference
log_file="save_dir/tvm_result.log"
touch $log_file
seq_lens="16 32 64 128 256 512"
for seq_len in $seq_lens
do
    # write log
    echo "============ TVM max_len = $seq_len ============" >> $log_file

    # prepare
    mkdir -p save_dir/tvm/$seq_len

    # tvm predict
    python3 tvm_convert.py --max_len $seq_len
    python3 tvm_predict.py --max_len $seq_len | grep infer >> $log_file
done
