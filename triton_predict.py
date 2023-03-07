import numpy as np
import tritonclient.http as httpclient
from utils import timer


def send_ids(triton_client, service_name, input_ids):
    inputs = []
    inputs.append(httpclient.InferInput('input_ids', input_ids.shape, "INT32"))
    inputs[0].set_data_from_numpy(input_ids, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('1537', binary_data=False))
    results = triton_client.infer(service_name, inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('1537')
    return output_data0


def triton_infer(triton_client, service_name, input_ids, times):
    outputs = []
    with timer("triton infer first time"):
        output = send_ids(triton_client, service_name, input_ids)
    with timer("triton infer {} times".format(times)):
        for _ in range(times):
            output = send_ids(triton_client, service_name, input_ids)
            outputs.append(output)
    return outputs


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
    service_name = "bert_openvino"
    times = 100
    max_len = 128
    input_ids = [i for i in range(3000, 3000 + max_len - 2)]
    input_ids = [101, *input_ids, 102]
    input_ids = np.array([input_ids]).astype(np.int32)
    triton_infer(triton_client, service_name, input_ids, times)
