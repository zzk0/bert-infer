import evaluate
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from utils import build_parser


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    parser = build_parser("tvm_predict")
    args = parser.parse_args()
    max_len = args.max_len

    model_path = "/home/percent1/models/nlp/text-classification/pretrained/roberta-base"
    save_path = "save_dir"
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(
            examples["text"], padding="max_length", max_length=max_len, truncation=True
        ),
        batched=True,
        batch_size=128,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        per_device_train_batch_size=128,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(os.path.join(save_path, "hf"))

