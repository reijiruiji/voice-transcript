import whisper
import sounddevice as sd
import numpy as np
import os
import torch
import scipy.io.wavfile as wav
from datetime import datetime
from pyannote.audio import Pipeline

# --- 設定 ---
# Hugging Faceで取得したトークンを環境変数 HF_TOKEN に設定してください
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODEL_SIZE = "base" 
SAMPLE_RATE = 16000
DURATION = 5  # 5秒ごとに解析

# フォルダ準備
if not os.path.exists("logs"):
    os.makedirs("logs")

log_file = f"logs/chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# モデルのロード
print("AIモデルをロード中（話者分離 + 文字起こし）...")
# GPUが使える場合は自動で割り当てられます
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
)
diarization_pipeline.to(device)

whisper_model = whisper.load_model(MODEL_SIZE, device=device)
print(f"使用デバイス: {device}")
print("準備完了。会話を開始してください（Ctrl+Cで終了）")

def start_chat_logging():
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"--- 録音開始: {datetime.now()} ---\n")
        try:
            while True:
                # 録音
                recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
                sd.wait()
                audio_data = recording.flatten().astype(np.float32)

                # 一時ファイルに保存（話者分離用）
                temp_file = "temp_segment.wav"
                wav.write(temp_file, SAMPLE_RATE, audio_data)
                
                # 1. 話者分離
                diarization = diarization_pipeline(temp_file, num_speakers=2)

                # 2. 文字起こし（多言語自動判別）
                result = whisper_model.transcribe(audio_data, fp16=(device.type == "cuda"))
                text = result['text'].strip()

                if text:
                    # この区間の主な話者を特定
                    speaker = "Speaker_?"
                    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                        speaker = speaker_label
                        break # 最初に見つかった話者を採用
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    # 簡易的に名前を割り当て
                    display_name = "あなた" if speaker == "SPEAKER_00" else "相手"
                    
                    output = f"[{timestamp}] {display_name}: {text}"
                    print(output)
                    f.write(output + "\n")
                    f.flush()

        except KeyboardInterrupt:
            print("\n終了します。ログを確認してください。")
            if os.path.exists("temp_segment.wav"):
                os.remove("temp_segment.wav")

if __name__ == "__main__":
    start_chat_logging()