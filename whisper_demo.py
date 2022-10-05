import whisper
import torch
import time

start_time = time.time()
import_time = time.time()

device = torch.device("cuda")

model = whisper.load_model("small", device=device)

print("--- %s seconds (import time) ---" % (time.time() - import_time))

audio_time = time.time()

audio = whisper.load_audio("dialog_1.mp3")
audio = whisper.pad_or_trim(audio)

print("--- %s seconds (audio time) ---" % (time.time() - audio_time))

mel_time = time.time()

mel = whisper.log_mel_spectrogram(audio).to(device)

print("--- %s seconds (get mel time) ---" % (time.time() - mel_time))

infer_time = time.time()

_, props = whisper.detect_language(model, mel)
print(f"Language: {max(props, key=props.get)}")

print("--- %s seconds (detect language time) ---" % (time.time() - infer_time))

decode_time = time.time()

options = whisper.DecodingOptions(fp16=True)
result = whisper.decode(model, mel, options)

print(result.text)

print("--- %s seconds (decode time) ---" % (time.time() - decode_time))
print("--- %s seconds (all executed time) ---" % (time.time() - start_time))
