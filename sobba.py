'''import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

def speak_to_microphone(api_key, region):
    print("Speak into your microphone. Say 'stop session' to end.")

    # Set up speech configuration
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # Enable speaker diarization
    speech_config.request_word_level_timestamps()  # For detailed transcription
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_SpeakerDiarizationMode, "True")

    # Create a speech recognizer with the given configurations
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    while True:
        speech_recognition_result = speech_recognizer.recognize_once_async().get()

        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"Recognized: {speech_recognition_result.text}")
            
            if "stop session" in speech_recognition_result.text.lower():
                print("Session ended by user.")
                break

            # Check if diarization is available
            if speech_recognition_result.speaker_id:
                print(f"Speaker: {speech_recognition_result.speaker_id}")
                
        elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
            print(f"No speech could be recognized: {speech_recognition_result.no_match_details}")
        elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_recognition_result.cancellation_details
            print(f"Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
                print("Did you set the speech resource key and region values?")

# Load environment variables
load_dotenv()

api_key = os.getenv("059a768e1d054aaaa51cc2becfe22fed")
region = os.getenv("eastus2")


speak_to_microphone(api_key, region)'''




'''import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

def speak_to_microphone(api_key, region):
    print("Speak into your microphone. Press Ctrl + C to end.")

    # Set up speech configuration
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # Enable detailed transcription (e.g., word-level timestamps)
    speech_config.request_word_level_timestamps()

    # Adjust timeout settings
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "60000"  # 60 seconds
    )
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "20000"  # 20 seconds
    )

    # Create a speech recognizer with the given configurations
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    speaker_labels = {}  # To keep track of speaker labels
    speaker_count = 0    # To track the number of unique speakers

    # Function to handle recognized speech in continuous mode
    def handle_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"Recognized: {evt.result.text}")
            
            # Check if diarization is available (Note: Check Azure docs for proper implementation)
            if hasattr(evt.result, "speaker_id") and evt.result.speaker_id is not None:
                speaker_id = evt.result.speaker_id
                # Assign a label if it's a new speaker
                if speaker_id not in speaker_labels:
                    nonlocal speaker_count
                    speaker_count += 1
                    speaker_labels[speaker_id] = f"Speaker {chr(64 + speaker_count)}"  # A, B, C...
                
                print(f"{speaker_labels[speaker_id]}: {evt.result.text}")

        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print(f"No speech could be recognized: {evt.result.no_match_details}")
        elif evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = evt.result.cancellation_details
            print(f"Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
                print("Did you set the speech resource key and region values?")

    # Attach event handlers for continuous recognition
    speech_recognizer.recognized.connect(handle_recognized)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition_async()

    try:
        while True:
            pass  # Keep the session active
    except KeyboardInterrupt:
        print("Session ended by user.")
        speech_recognizer.stop_continuous_recognition_async().get()  # Ensure recognition stops

# Load environment variables
load_dotenv()

api_key = os.getenv("AZURE_SPEECH_API_KEY")  # Replace with the actual environment variable name
region = os.getenv("AZURE_SPEECH_REGION")    # Replace with the actual environment variable name

speak_to_microphone(api_key, region)'''
import os
import threading
import queue
import azure.cognitiveservices.speech as speechsdk
import wave
import pyaudio
import tempfile
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Global queue to handle audio chunks for diarization
audio_queue = queue.Queue()

# Step 1: Real-Time Azure Speech-to-Text Transcription
def transcribe_audio_with_azure(api_key, region):
    # Set up Azure configuration
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # Create recognizer
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Event handler for recognized speech
    def recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"Recognized: {evt.result.text}")
            # Enqueue audio chunk for diarization
            audio_queue.put((evt.result.text, evt.result.offset))  # Store transcript with timestamp
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized.")
        elif evt.result.reason == speechsdk.ResultReason.Canceled:
            print(f"Speech recognition canceled: {evt.result.cancellation_details.reason}")

    # Connect event handler and start recognition
    speech_recognizer.recognized.connect(recognized)
    speech_recognizer.start_continuous_recognition_async()

    print("Transcription is running. Press Ctrl + C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Session ended by user.")
        speech_recognizer.stop_continuous_recognition_async().get()

# Step 2: Real-Time Speaker Diarization
def diarize_audio_with_pyannote(audio_chunk):
    # Load pre-trained speaker diarization model
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0",use_auth_token=True)

    # Create a temporary file to store the audio chunk for diarization
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        with wave.open(temp_audio_file.name, 'wb') as wf:
            wf.setnchannels(1)  # Mono channel
            wf.setsampwidth(2)  # Sample width (bytes)
            wf.setframerate(16000)  # Frame rate (Hz)

            # Write the audio chunk (this should be actual audio data in bytes)
            wf.writeframes(audio_chunk)

        # Apply the pipeline to the saved audio file to obtain speaker segments
        diarization = pipeline(temp_audio_file.name)

        # Extract speaker segments and labels
        speaker_info = []
        for segment, track, label in diarization.itertracks(yield_label=True):
            speaker_info.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": label
            })
    
    return speaker_info

# Step 3: Format Output
def format_output(transcripts, speaker_info):
    output = []
    for transcript, timestamp in transcripts:
        # Here we simply match the transcript to a speaker, assuming you have speaker timing info.
        # This would require actual logic to match speakers to transcripts based on timing.
        for info in speaker_info:
            if timestamp >= info["start"] and timestamp <= info["end"]:
                output.append(f"Speaker {info['speaker']}: {transcript}")

    return "\n".join(output)

# Example Usage: Capture Real-Time Audio Chunks
def main():
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("AZURE_SPEECH_API_KEY")  # Replace with actual Azure Speech API Key
    region = os.getenv("AZURE_SPEECH_REGION")    # Replace with actual Azure Speech Region

    # Example of how to handle audio capture
    def capture_audio_to_queue():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

        print("Capturing audio... Press Ctrl + C to stop.")

        try:
            while True:
                audio_data = stream.read(1024)
                # Enqueue audio data for diarization
                audio_queue.put(audio_data)
        except KeyboardInterrupt:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Audio capture ended.")

    # Run both Azure transcription and speaker diarization in parallel
    transcription_thread = threading.Thread(target=transcribe_audio_with_azure, args=(api_key, region))
    audio_capture_thread = threading.Thread(target=capture_audio_to_queue)

    transcription_thread.start()
    audio_capture_thread.start()

    # Continuously process audio chunks from the queue
    while True:
        if not audio_queue.empty():
            audio_chunk = audio_queue.get()
            speaker_info = diarize_audio_with_pyannote(audio_chunk)  # Process the audio chunk for diarization
            # Combine results (assuming you have collected all transcriptions)
            final_output = format_output(transcripts, speaker_info)
            print(final_output)  # Print combined results

if __name__ == "__main__":
    main()
