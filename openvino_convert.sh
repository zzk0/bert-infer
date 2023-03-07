# $1 indicates max_len
mo --input_model ./save_dir/onnx/$1/model.onnx --output_dir ./save_dir/ov/$1 --input_shape "(1, $1)"
