﻿# OpenAI-Whisper-Subtitle

# Automatic Video Subtitling with Whisper & FFmpeg

This script extracts audio from a video, transcribes and translates it to English using OpenAI's Whisper model, generates subtitles in SRT format, and then burns them onto the video using FFmpeg. If an NVIDIA GPU is available, NVENC acceleration is used for faster encoding.

## Features

- Extracts audio from a video file
- Uses OpenAI's Whisper model for transcription and translation
- Saves subtitles in SRT format
- Embeds subtitles into the video using FFmpeg (with optional NVENC acceleration)

## Requirements

Ensure you have the following installed:

- Python 3.7+
- [Whisper](https://github.com/openai/whisper)
- [MoviePy](https://zulko.github.io/moviepy/)
- [FFmpeg](https://ffmpeg.org/)
- [PyTorch](https://pytorch.org/) (with CUDA support if using GPU)

You can install dependencies using:

```bash
pip install whisper moviepy torch ffmpeg-python
```

## Usage

# 1. Prepare Your Video File

- Place your video file in the same directory as the script.
- Update the VIDEO_FILE variable with the filename.

# 2. Run the Script:

```bash
python script.py

```

## Process Explanation

- Extract Audio: Extracts audio from the video using MoviePy.
- Transcribe & Translate: Uses Whisper to transcribe and translate the audio to English.
- Generate Subtitles: Saves the transcript as an SRT subtitle file.
- Burn Subtitles: Embeds the subtitles onto the video using FFmpeg. If an NVIDIA GPU is available, NVENC acceleration is used for faster processing.

# What If You Don't Have an NVIDIA GPU?

- If you do not have an NVIDIA GPU, modify the FFmpeg command in the script to use a different encoder.
- Replace: vcodec="h264_nvenc" with vcodec="libx264"
- This will allow FFmpeg to use software encoding instead of GPU acceleration. However, encoding may take longer depending on your CPU.

# Output

- The final video with subtitles is saved as output_with_subtitles.mp4.

## Notes

- If no GPU is available, the script will automatically fall back to CPU for Whisper processing.
- Make sure ffmpeg is correctly installed and accessible via command line.

## License

- This script is open-source and free to use under the MIT License
