import whisper
import ffmpeg
import moviepy.editor as mp
import os
import torch

# --- CONFIG ---
VIDEO_FILE = "video.mp4"  # Change this to your video file
OUTPUT_VIDEO = "output_with_subtitles.mp4"
TEMP_AUDIO = "audio.wav"
SUBTITLES_FILE = "subtitles.srt"

# Step 1: Extract audio using MoviePy
print("Extracting audio...")
video = mp.VideoFileClip(VIDEO_FILE)

if video.audio:  # Check if audio exists
    video.audio.write_audiofile(TEMP_AUDIO, codec="pcm_s16le")
else:
    raise RuntimeError("No audio found in video!")

# Step 2: Transcribe & Translate using Whisper
print("Transcribing & translating audio...")

# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model("large-v2").to(device)  # Optimized model
result = model.transcribe(TEMP_AUDIO, task="translate")  # Translates to English

# Step 3: Save subtitles in SRT format
print("Saving subtitles...")
with open(SUBTITLES_FILE, "w", encoding="utf-8") as srt:
    for i, segment in enumerate(result["segments"]):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        srt.write(f"{i+1}\n")
        srt.write(f"{int(start // 3600):02}:{int(start % 3600 // 60):02}:{int(start % 60):02},{int((start % 1) * 1000):03} --> ")
        srt.write(f"{int(end // 3600):02}:{int(end % 3600 // 60):02}:{int(end % 60):02},{int((end % 1) * 1000):03}\n")
        srt.write(f"{text}\n\n")

print("Subtitles saved.")

# Step 4: Burn subtitles onto video using NVENC acceleration (only works with NVIDIA GPUs)
print("Burning subtitles with NVENC acceleration...")

# Convert subtitles path to absolute to avoid FFmpeg issues
absolute_subtitle_path = os.path.abspath(SUBTITLES_FILE)

ffmpeg.input(VIDEO_FILE).output(
    OUTPUT_VIDEO,
    vf=f"subtitles={SUBTITLES_FILE}:force_style='FontSize=24,PrimaryColour=&H00FFFF&'", #Set subtitle and size
    vcodec="h264_nvenc",  # Use NVENC for GPU acceleration
    preset="p4",  # Set Encoding Speed
    **{"b:v": "5M"},  # Specify video bitrate
    acodec="copy"  # Copy audio without re-encoding
).run(overwrite_output=True)




print(f"Done! Video with subtitles saved as {OUTPUT_VIDEO}")
