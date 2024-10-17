from pyannote.audio import Pipeline
import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import pipeline as hf_pipeline

# Load the pretrained diarization pipeline
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE"
)

# Send the pipeline to GPU (when available)
diarization_pipeline.to(torch.device("cpu"))

# Load the audio file
audio_file = "out.wav"

# Apply the pretrained pipeline
diarization = diarization_pipeline(audio_file)

# Step 1: Transcribe the audio
# Use Hugging Face's ASR model (you can also use your preferred speech-to-text model)
asr_model = hf_pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# Load audio file and preprocess it for the ASR model
audio_data, sample_rate = librosa.load(audio_file, sr=16000)

# Step 2: Perform speech recognition
# The model expects a file, so save the audio data as a temporary file
temp_audio_file = "out.wav"
sf.write(temp_audio_file, audio_data, sample_rate)

# Transcribe the audio
transcription_result = asr_model(temp_audio_file)
transcribed_text = transcription_result['text']

# Split the transcribed text into segments
# Here we will assume that the transcription is a single sentence. 
# In practice, you would need to split it into segments as you see fit.
transcribed_segments = transcribed_text.split('. ')  # You may change this based on your needs

# Step 3: Print the result with speaker labels and spoken text
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_time = turn.start
    end_time = turn.end

    # Find the corresponding transcribed text segment
    # Here we assume a simple mapping where each segment is spoken during the diarization period
    for segment in transcribed_segments:
        # This is a naive approach; you might want to implement a better matching logic based on time
        print(f"start={start_time:.1f}s stop={end_time:.1f}s speaker_{speaker}: {segment}")

# Clean up the temporary audio file
import os
os.remove(temp_audio_file)
