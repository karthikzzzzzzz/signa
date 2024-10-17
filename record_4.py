import whisper
from pyannote.audio import Pipeline, Audio
import warnings
"using whisper with labelled speakers"
# Ignore all warnings
warnings.filterwarnings("ignore")

# Load the speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=True)

# Load the Whisper model
model = whisper.load_model("medium.en")

# Load audio file and perform diarization
audio_file = "out.wav"
diarization_result = pipeline(audio_file)

# Initialize Audio for cropping segments
audio = Audio(sample_rate=16000, mono=True)

# Iterate through diarization results and transcribe each segment
for segment, _, speaker in diarization_result.itertracks(yield_label=True):
    # Crop the audio segment
    waveform, sample_rate = audio.crop(audio_file, segment)
    
    # Transcribe the cropped audio segment using Whisper
    text = model.transcribe(waveform.squeeze().numpy())["text"]
    
    # Print the results
    print(f"{segment.start:.2f}s {segment.end:.2f}s {speaker}: {text}")
