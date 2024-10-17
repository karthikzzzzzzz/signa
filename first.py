"not imp"

import whisper
from pyannote.audio import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Load the speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=True)

# Load the Whisper model
model = whisper.load_model("tiny.en")

# Function for transcription and diarization
def transcribe_and_diarize(audio_file):
    # Perform speaker diarization
    print("Performing speaker diarization...")
    diarization_result = pipeline(audio_file)
    print("Diarization complete.")

    # Transcribe the audio file using Whisper
    print("Transcribing audio...")
    asr_result = model.transcribe(audio_file)
    print("Transcription complete.")

    # Check the results
    print(f"Diarization Result: {diarization_result}")
    print(f"ASR Result: {asr_result}")

    # Collect results
    results = []

    for segment in diarization_result.itersegments():
        start_time = segment.start
        end_time = segment.end
        
        # Attempt to get the speaker label
        speaker = segment.label if hasattr(segment, 'label') else "Unknown"

        # Match segments with transcription
        for asr_segment in asr_result['segments']:
            if start_time >= asr_segment['start'] and end_time <= asr_segment['end']:
                transcript = asr_segment['text']
                results.append((start_time, end_time, speaker, transcript))
                break  # Break once you find the matching segment

    # Print results
    if results:
        for start, end, speaker, transcript in results:
            print(f"{start:.2f} {end:.2f} {speaker} {transcript}")
    else:
        print("No results found.")

# Example usage
transcribe_and_diarize("out.wav")
