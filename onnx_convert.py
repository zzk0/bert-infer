import os
import torch
from transformers import AutoModelForSequenceClassification
from utils import build_parser


def export_onnx(model, max_len, save_path):
    input_sample = [i for i in range(3000, 3000 + max_len - 2)]
    input_sample = [101, *input_sample, 102]
    input_sample = torch.Tensor([input_sample]).long()
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, input_sample, save_path)


if __name__ == '__main__':
    parser = build_parser("onnx_convert")
    args = parser.parse_args()
    max_len = args.max_len

    save_path = "save_dir"
    hf_save_path = os.path.join(save_path, "hf")
    hf_model = AutoModelForSequenceClassification.from_pretrained(hf_save_path, num_labels=2)
    export_onnx(hf_model, max_len, os.path.join(save_path, "onnx", str(max_len), "model.onnx"))

