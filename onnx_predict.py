import contextlib
import numpy as np
import onnx
import onnxruntime
import os
import time


@contextlib.contextmanager
def timer(message: str = ""):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(
            "{} - Elapsed time: {:.4f} ms".format(
                message, (end_time - start_time) * 1000
            )
        )


def load_onnx_sess(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    sess = onnxruntime.InferenceSession(model_path)
    return sess


def onnx_infer(ort_session, input_ids, times):
    outputs = []
    ort_inputs = {ort_session.get_inputs()[0].name: input_ids}
    with timer("onnx infer first time"):
        output = ort_session.run(None, ort_inputs)
    with timer("onnx infer {} times".format(times)):
        for _ in range(times):
            output = ort_session.run(None, ort_inputs)
            outputs.append(output)
    return outputs


times = 100
max_len = 128
save_path = "save_dir"
onnx_save_path = os.path.join(save_path, "onnx", "model.onnx")
ort_sess = load_onnx_sess(onnx_save_path)
input_sample = [i for i in range(3000, 3000 + max_len - 2)]
input_sample = [101, *input_sample, 102]
input_sample = np.array([input_sample])
outputs = onnx_infer(ort_sess, input_sample, times)
print(outputs[0])

