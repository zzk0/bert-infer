import os
import torch
from transformers import AutoModelForSequenceClassification


def export_onnx(model, max_len, save_path):
    input_sample = [i for i in range(3000, 3000 + max_len - 2)]
    input_sample = [101, *input_sample, 102]
    input_sample = torch.Tensor([input_sample]).long()
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, input_sample, save_path)


max_len = 128
save_path = "save_dir"
hf_save_path = os.path.join(save_path, "hf")
hf_model = AutoModelForSequenceClassification.from_pretrained(hf_save_path, num_labels=2)
export_onnx(hf_model, max_len, os.path.join(save_path, "onnx/model.onnx"))

