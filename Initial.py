import os
import time
import threading
import queue
import azure.cognitiveservices.speech as speechsdk
import pyaudio
import tempfile
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import warnings
import wave
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Global variables
audio_queue = queue.Queue()  # Initialize the audio queue
transcripts = []  # To keep track of transcripts
last_transcribed_text = ""  # Variable to track the last transcribed text

# Azure Speech SDK setup
def setup_speech_config():
    speech_key = os.getenv('AZURE_SPEECH_API_KEY')
    speech_region = os.getenv('AZURE_SPEECH_REGION')

    if not speech_key or not speech_region:
        print("Please set the environment variables SPEECH_KEY and SPEECH_REGION.")
        return None

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = "en-US"
    speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults, value='true')
    return speech_config

# Callbacks for transcription events
def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
    print('Canceled event')

def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStopped event')

def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    global last_transcribed_text  # Use the global variable

    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # Store the recognized text and its offset
        transcripts.append((evt.result.text, evt.result.offset, evt.result.speaker_id))  # Save transcript, timestamp, and speaker ID
        
        # Check if the recognized text is different from the last one
        if evt.result.text != last_transcribed_text:
            print('\tSpeaker ID({}):'.format(evt.result.speaker_id),evt.result.text)
            last_transcribed_text = evt.result.text  # Update the last recognized text
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        pass  # No action needed

def conversation_transcriber_transcribing_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    global last_transcribed_text  # Use the global variable

    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # Print interim results but only if it's different from the last recognized text
        if evt.result.text != last_transcribed_text:
            print('TRANSCRIBING:')
            print('\tText={}'.format(evt.result.text))
            print('\tSpeaker ID={}'.format(evt.result.speaker_id))

def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStarted event')

# Real-time audio capture
def capture_audio_to_queue():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    print("Capturing audio... Press Ctrl + C to stop.")

    try:
        while True:
            audio_data = stream.read(1024)
            audio_queue.put(audio_data)  # Enqueue audio data for diarization
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio capture ended.")

# Diarization function
def diarize_audio_with_pyannote(audio_chunk):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        with wave.open(temp_audio_file.name, 'wb') as wf:
            wf.setnchannels(1)  # Mono channel
            wf.setsampwidth(2)  # Sample width (bytes)
            wf.setframerate(16000)  # Frame rate (Hz)
            wf.writeframes(audio_chunk)  # Write audio data to temp file

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

# Format and print output
def format_output(transcripts, speaker_info):
    output = []
    for transcript, timestamp, speaker_id in transcripts:
        # Match the transcript with the speaker based on timing
        for info in speaker_info:
            if timestamp >= info["start"] and timestamp <= info["end"]:
                output.append(f"Speaker {info['speaker']:02}: {transcript}")
    return "\n".join(output)

# Main function to run transcription and diarization
def main():
    speech_config = setup_speech_config()
    if not speech_config:
        return

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)

    transcribing_stop = False

    # Define stop callback
    def stop_cb(evt: speechsdk.SessionEventArgs):
        print('CLOSING on {}'.format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True

    # Connect callbacks to the events fired by the conversation transcriber
    conversation_transcriber.transcribed.connect(conversation_transcriber_transcribed_cb)
    conversation_transcriber.transcribing.connect(conversation_transcriber_transcribing_cb)
    conversation_transcriber.session_started.connect(conversation_transcriber_session_started_cb)
    conversation_transcriber.session_stopped.connect(conversation_transcriber_session_stopped_cb)
    conversation_transcriber.canceled.connect(conversation_transcriber_recognition_canceled_cb)
    
    # Stop transcribing on either session stopped or canceled events
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    # Start the transcription
    conversation_transcriber.start_transcribing_async()

    # Start audio capture in a separate thread
    audio_capture_thread = threading.Thread(target=capture_audio_to_queue)
    audio_capture_thread.start()

    # Continuously process audio chunks from the queue
    while not transcribing_stop:
        if not audio_queue.empty():
            audio_chunk = audio_queue.get()
            speaker_info = diarize_audio_with_pyannote(audio_chunk)  # Process the audio chunk for diarization
            # Combine results and print
            final_output = format_output(transcripts, speaker_info)
            if final_output:  # Print only if there's output
                print(final_output)  # Print combined results

    conversation_transcriber.stop_transcribing_async()

# Main execution
if __name__ == "__main__":
    main()
