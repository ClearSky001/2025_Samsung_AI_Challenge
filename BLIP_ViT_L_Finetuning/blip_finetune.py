import os
import json
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List
from PIL import Image

from transformers import (
    BlipProcessor, 
    BlipForQuestionAnswering,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from torch.utils.data import Dataset

# 하이퍼파라미터 로드
with open("/content/drive/MyDrive/Colab Notebooks/2025_Samsung_AI_Challenge/optuna_best_params_final.json", "r") as f:
    best_params = json.load(f)["best_params"]

# VQA 데이터셋 정의
@dataclass
class VQADataset(Dataset):
    data: List[Dict]
    processor: BlipProcessor

    def __getitem__(self, idx):
        sample = self.data[idx]

        image_path = sample["image_path"]
        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=sample["question"],
            padding="max_length",
            max_length=32,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        encoding["labels"] = self.processor.tokenizer(
            sample["answer"],
            padding="max_length",
            truncation=True,
            max_length=10,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        return encoding

    def __len__(self):
        return len(self.data)

# Metrics 관련 함수들
def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_preds = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    acc = np.mean([pred.strip().lower() == label.strip().lower() 
                   for pred, label in zip(decoded_preds, decoded_labels)])
    return {"text_exact_match": acc, "eval_samples": len(predictions)}

# Processor 및 모델 로드
model_name = "Salesforce/blip-vqa-capfilt-large"
processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
model = BlipForQuestionAnswering.from_pretrained(model_name)

# 실제 Train/Validation 데이터 로드
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

train_data_path = "/content/drive/MyDrive/Colab Notebooks/2025_Samsung_AI_Challenge/dataset/VQAv2/train_fixed.json"
val_data_path = "/content/drive/MyDrive/Colab Notebooks/2025_Samsung_AI_Challenge/dataset/VQAv2/val_fixed.json"

train_data = load_json(train_data_path)
val_data = load_json(val_data_path)

train_dataset = VQADataset(train_data, processor)
val_dataset = VQADataset(val_data, processor)

# 학습 설정
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/Colab Notebooks/2025_Samsung_AI_Challenge/blip_final_model",
    num_train_epochs=best_params.get("num_train_epochs", 5),
    per_device_train_batch_size=best_params.get("per_device_train_batch_size", 16),
    per_device_eval_batch_size=best_params.get("per_device_eval_batch_size", 16),
    learning_rate=best_params.get("learning_rate", 5e-5),
    weight_decay=best_params.get("weight_decay", 0.01),
    warmup_ratio=best_params.get("warmup_ratio", 0.1),
    lr_scheduler_type=best_params.get("lr_scheduler_type", "linear"),
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="text_exact_match",
    greater_is_better=True,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    eval_accumulation_steps=8
)

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 학습 시작
trainer.train()

# generate 기반 평가 (옵션)
def evaluate_with_generation(model, dataloader, processor, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating..."):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=20
            )
            preds = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            golds = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            all_preds.extend(preds)
            all_labels.extend(golds)

    acc = np.mean([
        p.strip().lower() == g.strip().lower() 
        for p, g in zip(all_preds, all_labels)
    ])
    print(f"Eval (generate-based) Accuracy: {acc:.4f}")
