import os
import threading
import queue
import azure.cognitiveservices.speech as speechsdk
from datetime import datetime
import openai
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import tempfile
import warnings
import wave
import keyboard  # Keep this if you want to use keyboard input for stopping

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Global variables
audio_queue = queue.Queue()  # Initialize the audio queue
transcripts = []  # To keep track of transcripts
last_transcribed_text = ""  # Variable to track the last transcribed text

# Azure Speech SDK setup
def setup_speech_config():
    # Define the list of languages you want to support for auto-detection
    languages = [
        "en-US", "fr-FR", "es-ES", "de-DE", "hi-IN",
        "zh-CN", "ja-JP", "ru-RU", "it-IT", "ar-EG"
    ]
    
    speech_key = os.getenv('AZURE_SPEECH_API_KEY')
    speech_region = os.getenv('AZURE_SPEECH_REGION')

    if not speech_key or not speech_region:
        print("Please set the environment variables AZURE_SPEECH_API_KEY and AZURE_SPEECH_REGION.")
        return None

    # Create the base speech configuration
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)

    # Use AutoDetectSourceLanguageConfig to allow automatic language detection
    auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)

    # Set properties for speaker diarization
    speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults, value='true')
    
    return speech_config, auto_detect_config

# Callbacks for transcription events
def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
    print('Canceled event')

def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStopped event')

def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    global last_transcribed_text  # Use the global variable

    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        pc_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        transcripts.append((evt.result.text, pc_time, evt.result.speaker_id))
        
        if evt.result.text != last_transcribed_text:
            print(f'[{pc_time}] Speaker ID({evt.result.speaker_id}): {evt.result.text}')
            last_transcribed_text = evt.result.text  # Update the last recognized text
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        pass  # No action needed

def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStarted event')

# Diarization function (remains unchanged)
def diarize_audio_with_pyannote(audio_chunk):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)
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
        for segment, label in diarization.itertracks(yield_label=True):
            speaker_info.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": label
            })
    
    return speaker_info

# Format and print output (remains unchanged)
def format_output(transcripts, speaker_info):
    output = []
    for transcript, timestamp, speaker_id in transcripts:
        for info in speaker_info:
            if timestamp >= info["start"] and timestamp <= info["end"]:
                output.append(f"Speaker {info['speaker']:02}: {transcript}")
    return "\n".join(output)

# Function to summarize transcriptions using OpenAI API (remains unchanged)
def summarize_transcriptions(transcripts):
    combined_transcript = "\n".join([f"Speaker {speaker_id}: {text}" for text, _, speaker_id in transcripts])
    openai.api_key = os.getenv('OPEN_AI_API_KEY')
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Or use another model like "gpt-4" if available
            messages=[
                {"role": "user", "content": f"Please summarize the following conversation:\n\n{combined_transcript}"}
            ]
        )
        summary = response['choices'][0]['message']['content']
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None

# Main function to run transcription and diarization
def main():
    speech_config, auto_detect_config = setup_speech_config()  # Get both configs
    if not speech_config:
        return

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # Create ConversationTranscriber with automatic language detection
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config, 
        audio_config=audio_config
    )
    
    # Set the auto-detect language configuration
    conversation_transcriber.source_language_config = auto_detect_config

    transcribing_stop = False

    # Define stop callback
    def stop_cb(evt: speechsdk.SessionEventArgs):
        print('CLOSING on {}'.format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True

    # Connect callbacks to the events fired by the conversation transcriber
    conversation_transcriber.transcribed.connect(conversation_transcriber_transcribed_cb)
    conversation_transcriber.session_started.connect(conversation_transcriber_session_started_cb)
    conversation_transcriber.session_stopped.connect(conversation_transcriber_session_stopped_cb)
    conversation_transcriber.canceled.connect(conversation_transcriber_recognition_canceled_cb)

    # Stop transcribing on either session stopped or canceled events
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    # Start the transcription with automatic language detection
    conversation_transcriber.start_transcribing_async()

    print("Recording... Press Enter to stop.")

    # Wait for user input to stop
    input("Press Enter to stop...")  # Replace keyboard wait with input

    # Stop the Azure Speech SDK transcription
    conversation_transcriber.stop_transcribing_async().get()  # Wait for the stop to complete

    # Process transcripts
    print(transcripts)
    summary = summarize_transcriptions(transcripts)
    if summary:
        print("\nSummary of the conversation:")
        print(summary)


# Main execution
if __name__ == "__main__":
    main()
