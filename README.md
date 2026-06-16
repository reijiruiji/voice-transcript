# VoiceTranscript - リアルタイム音声文字起こし

Whisper + pyannote.audioを使ったリアルタイム音声文字起こし・話者分離ツールです。

## 機能

- **リアルタイム文字起こし** — OpenAI Whisperで5秒ごとに音声を解析
- **話者分離** — pyannote.audioで「あなた」「相手」を自動判別
- **GPU対応** — CUDA利用可能時は自動でGPUを使用
- **ログ保存** — 会話内容をタイムスタンプ付きでテキスト保存

## セットアップ

### 必要なもの

- Python 3.x
- [Hugging Face](https://huggingface.co/)アカウントと `pyannote/speaker-diarization-3.1` へのアクセス許可

### インストール

```bash
pip install openai-whisper sounddevice numpy torch scipy pyannote.audio
```

### 環境変数の設定

```bash
# .envファイルまたは環境変数に設定
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

### 実行

```bash
cd my_transcriber/my_transcriber
python main.py
# Ctrl+C で終了
```

ログは `logs/` フォルダに保存されます。
