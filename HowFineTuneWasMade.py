import os
import re
import time
import numpy as np
import pandas as pd
import librosa
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback
)
from sklearn.model_selection import train_test_split

# =========================
# CONFIGURATION
# =========================
BASE_MODEL = os.environ.get("BASE_MODEL_PATH", "openai/whisper-small")

BASE_DIR = Path(__file__).resolve().parent
dataset_folder = os.environ.get("DATASET_NAME", "Medicines")

if "DATASET_FULL_PATH" in os.environ:
    DATA_DIR = Path(os.environ["DATASET_FULL_PATH"])
    print(f"🚀 [RAM MODE] Loading data from memory: {DATA_DIR}")
else:
    DATA_DIR = BASE_DIR / "Create_Set_API" / dataset_folder
    print(f"🐢 [DISK MODE] Loading data from disk: {DATA_DIR}")

OUTPUT_ROOT_DIR = Path(os.environ.get("OUTPUT_ROOT_DIR", BASE_DIR))

SAMPLE_RATE = 16000
LEARNING_RATE = 1e-5
NUM_EPOCHS = int(os.environ.get("TRAINING_EPOCHS", 3))

BATCH_PER_GPU = 2
GRADIENT_ACCUMULATION = 32

# =========================
# 1. HELPER FUNCTIONS
# =========================
def get_rows_from_transcript(txt_path: str, dtype: str) -> List[Dict[str, str]]:
    rows = []
    if not os.path.exists(txt_path):
        return rows
    med_dir = os.path.basename(os.path.dirname(txt_path))
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            fname, text = line.split("|", 1)
            wav_path = os.path.join(os.path.dirname(txt_path), fname)
            if os.path.exists(wav_path):
                rows.append({
                    "path": wav_path,
                    "sentence": text,
                    "dtype": dtype,
                    "medicine": med_dir
                })
    return rows


# =========================
# 2. DATASET (WITH DURATION CALCULATION)
# =========================
class MedicinesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: WhisperProcessor, sample_rate: int = 16000):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["path"]
        text = row["sentence"]

        MAX_SAMPLES = 480_000

        # Calculating actual duration before padding
        if audio_path == "<GENERATED_SILENCE>":
            audio = np.zeros(MAX_SAMPLES, dtype=np.float32)
            duration = 0.0
        elif audio_path == "<GENERATED_NOISE>":
            noise = np.random.normal(0, 0.005, MAX_SAMPLES)
            audio = np.asarray(noise, dtype=np.float32)
            duration = 0.0
        else:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio) / self.sample_rate

        if len(audio) == 0:
             audio = np.zeros(MAX_SAMPLES, dtype=np.float32)
             duration = 0.0

        return {
            "audio": np.asarray(audio, dtype=np.float32),
            "sentence": text,
            "duration": duration  # Passing duration to the collator
        }


# =========================
# 3. COLLATOR (TIMESTAMP INJECTION)
# =========================
@dataclass
class SimpleWhisperCollator:
    processor: WhisperProcessor
    sampling_rate: int = 16000
    max_samples: int = 480_000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [f["audio"] for f in features]

        # Preparing labels with timestamps
        new_labels = []
        for f in features:
            text = f["sentence"].strip()
            duration = f["duration"]

            if duration == 0:
                # Negative samples (silence/noise) get an empty label (only EOT)
                new_labels.append("")
            else:
                # Whisper quantizes time every 0.02s. Rounding duration.
                rounded_duration = round(duration / 0.02) * 0.02
                # Failsafe to ensure duration doesn't exceed 30.00 seconds
                rounded_duration = min(rounded_duration, 30.00)

                # Format: <|0.00|> Text <|time|>
                new_labels.append(f"<|0.00|>{text}<|{rounded_duration:.2f}|>")

        inputs = self.processor(
            audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
            return_attention_mask=True
        )

        labels = self.processor.tokenizer(
            new_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True  # Required for correct tag parsing
        ).input_ids

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return {
            "input_features": inputs.input_features,
            "attention_mask": inputs.attention_mask,
            "labels": labels
        }


# =========================
# 4. TIMING CALLBACK
# =========================
class EpochTimingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is not None:
            duration = time.time() - self.epoch_start_time
            self.epoch_times.append(duration)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            print("\n" + "="*40)
            print("⏱️  TRAINING TIME SUMMARY")
            print("="*40)
            for i, duration in enumerate(self.epoch_times, 1):
                minutes = duration / 60
                print(f"✅ Epoch {i}: {duration:.2f} s ({minutes:.2f} min)")
            print("="*40 + "\n")


# =========================
# 5. MAIN
# =========================
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    is_main_process = (local_rank in [-1, 0])

    seed_val = int(os.environ.get("TRAINING_SEED", 42))
    dir_name = f"{dataset_folder}_{seed_val}"
    output_dir = OUTPUT_ROOT_DIR / dir_name

    if is_main_process:
        print(f"👋 Starting training. DDP Rank: {local_rank}")
        os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, "dataset.csv")

    if is_main_process:
        if not os.path.exists(csv_path):
            print("📄 Generating dataset.csv...")
            all_rows = []
            if not os.path.exists(DATA_DIR):
                raise FileNotFoundError(f"❌ No data found in {DATA_DIR}")

            for medicine in os.listdir(DATA_DIR):
                med_dir = os.path.join(DATA_DIR, medicine)
                if not os.path.isdir(med_dir):
                    continue
                all_rows += get_rows_from_transcript(os.path.join(med_dir, "transcript.txt"), "word")
                all_rows += get_rows_from_transcript(os.path.join(med_dir, "transcript_sentences.txt"), "sentence")

            df_build = pd.DataFrame(all_rows).drop_duplicates(subset=["path"])
            df_build.to_csv(csv_path, index=False)
        else:
            print(f"📄 Using existing CSV: {csv_path}")

    if local_rank != -1:
        dist.barrier()

    df = pd.read_csv(csv_path)

    if "dtype" not in df.columns: df["dtype"] = "word"
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["dtype"], random_state=seed_val)

    def add_negative_samples(df_in, silence_percent=0.0, noise_percent=0.0):
        original_len = len(df_in)
        n_silence = int(original_len * silence_percent)
        n_noise = int(original_len * noise_percent)
        new_rows = []

        if n_silence > 0:
            for _ in range(n_silence):
                new_rows.append({
                    "path": "<GENERATED_SILENCE>",
                    "sentence": "",
                    "dtype": "negative_silence"
                })

        if n_noise > 0:
            for _ in range(n_noise):
                new_rows.append({
                    "path": "<GENERATED_NOISE>",
                    "sentence": "",
                    "dtype": "negative_noise"
                })

        if not new_rows:
            return df_in
        return pd.concat([df_in, pd.DataFrame(new_rows)], ignore_index=True)

    if is_main_process:
        print(f"🔇 Applying Negative Sampling (Train: +10% silence / +10% noise, Val: +10% / +10%)...")

    # Values changed from 0.30/0.20 to 0.10/0.10
    train_df = add_negative_samples(train_df, silence_percent=0.10, noise_percent=0.10)
    val_df = add_negative_samples(val_df, silence_percent=0.10, noise_percent=0.10)

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="pl", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    model.get_encoder().requires_grad_(False)

    # Enable timestamp generation
    model.config.predict_timestamps = True
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="pl",
        task="transcribe",
        no_timestamps=False  # Unlocking timestamps
    )
    model.config.condition_on_prev_tokens = False

    train_dataset = MedicinesDataset(train_df, processor, SAMPLE_RATE)
    val_dataset = MedicinesDataset(val_df, processor, SAMPLE_RATE)
    data_collator = SimpleWhisperCollator(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        seed=seed_val,
        data_seed=seed_val,
        per_device_train_batch_size=BATCH_PER_GPU,
        per_device_eval_batch_size=BATCH_PER_GPU,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        gradient_checkpointing=False,
        fp16=True,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=500,
        weight_decay=0.01,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor,
        callbacks=[EpochTimingCallback()]
    )

    if is_main_process:
        print("🚀 Starting training...")

    trainer.train()

    if local_rank != -1:
        dist.barrier()

    if is_main_process:
        print(f"✅ Saving model to: {output_dir}")
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))

        tracker_file = OUTPUT_ROOT_DIR / "LastModel.txt"
        try:
            with open(tracker_file, "w") as f:
                f.write(str(output_dir))
            print(f"📝 Updated tracker: {tracker_file} -> {output_dir}")
        except Exception as e:
            print(f"⚠️ Error updating LastModel.txt: {e}")

    if local_rank != -1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()