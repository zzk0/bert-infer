import contextlib
import os
import torch
import time
from transformers import AutoModelForSequenceClassification


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


def torch_infer(model, input_ids, times):
    model.eval()
    outputs = []
    with torch.no_grad():
        with timer("torch infer first time"):
            output = model(input_ids)
        with timer("torch infer {} times".format(times)):
            for _ in range(times):
                output = model(input_ids)
                outputs.append(output)
    return outputs


times = 100
max_len = 128
save_path = "save_dir"
hf_save_path = os.path.join(save_path, "hf")
model = AutoModelForSequenceClassification.from_pretrained(hf_save_path, num_labels=2)
input_sample = [i for i in range(3000, 3000 + max_len - 2)]
input_sample = [101, *input_sample, 102]
input_sample = torch.Tensor([input_sample]).long()
outputs = torch_infer(model, input_sample, times)
print(outputs[0])
