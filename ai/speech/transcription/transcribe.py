import os, math, csv
import soundfile
import torch
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Specify Paths
base_path = "/home/kampff/Dropbox/Voight-Kampff/NoBlackBoxes/LastBlackBox/videos/manifesto"
input_path = f"{base_path}/audio_16k.wav"    # 16 kHz mono S16 WAV
output_csv = f"{input_path[:-4]}.csv"        # CSV transcription
output_txt = f"{input_path[:-4]}.txt"        # Text transcription

# Specify Params
sample_rate = 16000                         # Sample Rate (Hz)
whisper_model = "openai/whisper-medium"     # Good compromise (for CPU inference)
language = "en"                             # English
task = "transcribe"                         # or "translate" if you want English from any language
chunk_length_s = 25.0                       # seconds per chunk
chunk_overlap_s = 1.0                       # seconds of overlap between chunks
max_new_tokens = 224                        # cap tokens per chunk to avoid rambling
no_repeat_ngram_size = 5                    # reduce repeated loops

# Detect CUDA
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

# Load Model
print(f"Loading Whisper Model: {whisper_model}")
processor = WhisperProcessor.from_pretrained(whisper_model)
model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
model.to(device)
model.eval()
if hasattr(model, "generation_config"):
    model.generation_config.forced_decoder_ids = None
if hasattr(model, "config"):
    model.config.forced_decoder_ids = None

# Load Audio
print(f"Loading Audio: {os.path.basename(input_path)}")
audio, sr = soundfile.read(input_path)
if sr != sample_rate: # Valid sample rate (must be 16 kHz)
    raise ValueError(f"Expected {sample_rate} Hz audio, got {sr} Hz. Convert with ffmpeg first.")
if audio.ndim > 1: # Mono
    audio = audio.mean(axis=1)
total_samples = audio.shape[0]
total_duration_s = total_samples / sample_rate
chunk_samples = int(chunk_length_s * sample_rate)
overlap_samples = int(chunk_overlap_s * sample_rate)
step_samples = chunk_samples - overlap_samples
num_chunks = math.ceil(max(1, (total_samples - overlap_samples) / step_samples))
print(f"- duration: {total_duration_s:.1f} s")
print(f"- num_chunks: {num_chunks}")

# ------------------
# Transcription Loop
# ------------------
print(f"Transcribing...")
segments = []
current_chunk = 0
with torch.no_grad():
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        current_chunk += 1
        progress = (start / total_samples) * 100.0
        print(f"\nProcessing Chunk ({current_chunk}/{num_chunks}): ({progress:.1f}%): samples {start} -> {end}")

        # Prepare model inputs
        inputs = processor(
            chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # The attention mask may not be required
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Generate text for this chunk
        generated_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        # Decode text for this chunk
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Approximate timings for this chunk
        start_sec = start / sample_rate
        end_sec = end / sample_rate

        # Save segment
        segments.append(
            {
                "text": text,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_sample": int(start),
                "end_sample": int(end),
            }
        )

        # Print text for feedback
        print(f"[{start_sec:.2f} - {end_sec:.2f}] {text}")

        # Move to next chunk (with overlap)
        if end == total_samples:
            break
        start += step_samples

# Merge next chunk's text (remove overlap/duplicates) with previous
tail_window = 30        # How many words to consider in next chunk (tail)
merged = [segments[0]]  # Store first chunk
for next in segments[1:]:
    prev_words = merged[-1]["text"].split()
    next_words = next["text"].split()
    tail = prev_words[-tail_window:]
    max_k = min(len(tail), len(next_words))
    k = 0
    for n in range(1, max_k + 1):               # Find longest overlap k
        if tail[-n:] == next_words[:n]:
            k = n
    if k > 0:
        next["text"] = " ".join(next_words[k:]).strip() # Trim overlap
        print(f"Trimmed: {k}")
    merged.append(next)
segments = merged

# Save Transcription to CSV file
print(f"Saving transcript to: {os.path.basename(output_csv)}")
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["start_sec", "end_sec", "start_sample", "end_sample", "text"]
    )
    for seg in segments:
        writer.writerow(
            [
                seg["start_sec"],
                seg["end_sec"],
                seg["start_sample"],
                seg["end_sample"],
                seg["text"],
            ]
        )

# Save Transcription to TXT file
print(f"Saving transcript to: {os.path.basename(output_txt)}")
with open(output_txt, "w", encoding="utf-8") as f:
    for seg in segments:
        f.write(seg["text"] + "\n")

# FIN