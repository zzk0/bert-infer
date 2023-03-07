import contextlib
import os
import torch
import time
from transformers import AutoModelForSequenceClassification
from utils import timer, build_parser


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


if __name__ == '__main__':
    parser = build_parser("torch_predict")
    args = parser.parse_args()
    max_len = args.max_len
    times = args.times

    save_path = "save_dir"
    hf_save_path = os.path.join(save_path, "hf")
    model = AutoModelForSequenceClassification.from_pretrained(hf_save_path, num_labels=2)
    input_sample = [i for i in range(3000, 3000 + max_len - 2)]
    input_sample = [101, *input_sample, 102]
    input_sample = torch.Tensor([input_sample]).long()
    outputs = torch_infer(model, input_sample, times)
    print(outputs[0])
