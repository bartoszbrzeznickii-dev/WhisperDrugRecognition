# -*- coding: utf-8 -*-
import os
import re
import csv
import time
import sys
import gc
import zlib  # Added for compression handling
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# =========================
# CONFIGURATION (ARGPARSE)
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Validation Pipeline")

    parser.add_argument("--model_path", type=str, required=True, help="Full path to the model folder")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the main folder with audio data")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder where to save the resulting CSV")
    parser.add_argument("--folders", nargs='+', required=True, help="List of subfolders to process (e.g. 200_1_Office ...)")

    return parser.parse_args()

# =========================
# CONSTANTS
# =========================
LANG = "pl"
TASK = "transcribe"
SAMPLE_RATE = 16000
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

# =========================
# HELPER FUNCTIONS
# =========================
def norm_text_simple(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\wąćęłńóśżź ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def word_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(text or "")]

def trim_hypothesis_by_ref_last_words(hyp: str, ref: str) -> str:
    hyp_str = hyp or ""
    ref_norm = norm_text_simple(ref or "")
    ref_tokens = ref_norm.split()
    if not hyp_str or not ref_tokens:
        return hyp_str.strip()
    last = ref_tokens[-1]
    prev = ref_tokens[-2] if len(ref_tokens) >= 2 else None
    last_count_in_ref = sum(1 for t in ref_tokens if t == last)
    hspans = word_spans(hyp_str)
    seen = 0
    for w, s, e in hspans:
        if w == last:
            seen += 1
            if seen == last_count_in_ref:
                return hyp_str[:e].strip()
    if prev is not None:
        prev_positions = [(idx, s, e) for idx, (w, s, e) in enumerate(hspans) if w == prev]
        if prev_positions:
            last_idx, s, e = prev_positions[-1]
            if last_idx + 1 < len(hspans):
                _, _, e_next = hspans[last_idx + 1]
                return hyp_str[:e_next].strip()
            else:
                return hyp_str[:e].strip()
    return hyp_str.strip()

def load_audio(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros(16000, dtype="float32")

def read_transcript(txt_path: Path) -> List[Tuple[str, str]]:
    rows = []
    if not txt_path.exists():
        return rows
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            if "|" in ln:
                fname, ref = ln.strip().split("|", 1)
                rows.append((fname.strip(), ref.strip()))
    return rows

def collect_pairs(dataset_dir: Path) -> List[Tuple[str, Path, str, str]]:
    out = []
    if not dataset_dir.exists():
        print(f"WARNING: Directory {dataset_dir} does not exist!")
        return out
    for drug_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        drug = drug_dir.name.replace("_", " ")
        txt_files = list(drug_dir.glob("*.txt"))
        if txt_files:
            pairs = read_transcript(txt_files[0])
            for fname, ref in pairs:
                wav_p = drug_dir / fname
                if wav_p.exists():
                    out.append((drug, wav_p, fname, ref))
    return out

def get_compression_ratio(text: str) -> float:
    if not text:
        return 0.0
    text_bytes = text.encode("utf-8")
    compressed = zlib.compress(text_bytes)
    return len(text_bytes) / len(compressed)

class WhisperASR:
    def __init__(self, model_dir: Path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--> Loading model from: {model_dir}")
        print(f"--> Device: {self.device}")
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(str(model_dir))

        # --- TRANSFORMERS LIBRARY BUG FIX ---
        if isinstance(self.model.config.eos_token_id, list):
            self.model.config.eos_token_id = self.model.config.eos_token_id[0]
        if hasattr(self.model, "generation_config") and isinstance(self.model.generation_config.eos_token_id, list):
            self.model.generation_config.eos_token_id = self.model.generation_config.eos_token_id[0]
        # ----------------------------------------------

        self.model.to(self.device).eval()

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
        feats = inputs.input_features.to(self.device)
        attn_mask = inputs.attention_mask.to(self.device)

        # Standard temperature fallback according to OpenAI guidelines
        temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        best_text = ""

        for temp in temperatures:
            do_sample = temp > 0.0

            gen_kwargs = {
                "max_new_tokens": 256,
                "language": LANG,
                "task": TASK,
                "return_timestamps": True,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = temp

            pred_ids = self.model.generate(
                feats,
                attention_mask=attn_mask,
                **gen_kwargs
            )

            text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

            # Empty text means 0.0 compression, break the loop and return it
            if not text:
                return text

            compression_ratio = get_compression_ratio(text)

            # If the ratio is less than or equal to 2.4, the result is acceptable
            if compression_ratio <= 2.4:
                return text

            # If the threshold is exceeded, the loop will move to the next (higher) temperature
            best_text = text

        # Return the last result if the compression threshold is still exceeded at the maximum temperature
        return best_text

def process_single_model(args, all_pairs):
    model_path = Path(args.model_path)
    model_name = model_path.name
    output_root = Path(args.output_dir)

    print(f"\n{'='*60}")
    print(f"🚀 STARTING WORK ON MODEL: {model_name}")
    print(f"{'='*60}")

    output_csv_path = output_root / f"{model_name}.csv"

    if not model_path.exists():
        print(f"❌ ERROR: Model directory does not exist: {model_path}")
        return

    try:
        asr = WhisperASR(model_path)
    except Exception as e:
        print(f"❌ ERROR while loading model {model_name}: {e}")
        return

    results = []
    t_start = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Starting transcription of {len(all_pairs)} files...")

    for i, (ds_name, drug, wav_p, fname, ref) in enumerate(all_pairs, 1):
        if i % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"   [Model: {model_name}] Progress: {i}/{len(all_pairs)} (time: {elapsed:.1f}s)")

        audio = load_audio(wav_p)
        raw_text = asr.transcribe(audio)
        hyp_text = trim_hypothesis_by_ref_last_words(raw_text, ref)

        results.append({
            "dataset": ds_name,
            "drug": drug,
            "filename": fname,
            "filepath": str(wav_p),
            "reference": ref,
            "hypothesis": hyp_text
        })

    print(f"💾 Saving results to: {output_csv_path}")
    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["dataset", "drug", "filename", "filepath", "reference", "hypothesis"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(results)

    total_time = time.perf_counter() - t_start
    print(f"✅ Finished model {model_name} in {total_time:.2f}s")
    del asr
    del results
    gc.collect()
    torch.cuda.empty_cache()

def main():
    args = parse_args()
    print("=== INFERENCE LOOP SCRIPT START (ARGPARSE) ===\n")

    data_root = Path(args.data_root)
    all_pairs = []

    print(f"Main data path: {data_root}")
    print(f"Folders to process: {args.folders}")

    for folder_name in args.folders:
        folder_path = data_root / folder_name
        print(f"--> Fetching files from: {folder_path}")
        current_pairs = collect_pairs(folder_path)
        for p in current_pairs:
            all_pairs.append((folder_name, *p))

    if not all_pairs:
        print("❌ ERROR: No audio files found in the specified folders!")
        sys.exit(1)

    print(f"Total audio files found: {len(all_pairs)}")
    process_single_model(args, all_pairs)
    print("=== TASK COMPLETED ===")

if __name__ == "__main__":
    main()