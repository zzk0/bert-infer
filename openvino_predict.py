import numpy as np
import openvino.runtime as ov
import os
from utils import timer, build_parser


def load_ov_model(model_path):
    core = ov.Core()
    compiled_model = core.compile_model(model_path, "AUTO:CPU")
    return compiled_model


def ov_infer(ov_model, input_ids, times):
    outputs = []
    with timer("ov infer first time"):
        infer_request = ov_model.create_infer_request()
        input_tensor = ov.Tensor(input_ids)
        infer_request.set_input_tensor(input_tensor)
        infer_request.start_async()
        infer_request.wait()
        output = infer_request.get_output_tensor()
        output_buffer = output.data
    with timer("ov infer {} times".format(times)):
        for _ in range(times):
            infer_request = ov_model.create_infer_request()
            input_tensor = ov.Tensor(input_ids)
            infer_request.set_input_tensor(input_tensor)
            infer_request.start_async()
            infer_request.wait()
            output = infer_request.get_output_tensor()
            output_buffer = output.data
            outputs.append(output_buffer)
    return outputs



if __name__ == '__main__':
    parser = build_parser("openvino_predict")
    args = parser.parse_args()
    max_len = args.max_len
    times = args.times

    save_path = "save_dir"
    onnx_save_path = os.path.join(save_path, "ov", str(max_len), "model.xml")
    ov_model = load_ov_model(onnx_save_path)
    input_sample = [i for i in range(3000, 3000 + max_len - 2)]
    input_sample = [101, *input_sample, 102]
    input_sample = np.array([input_sample])
    outputs = ov_infer(ov_model, input_sample, times)
    print(outputs[0])

