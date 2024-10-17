import os
import queue
import azure.cognitiveservices.speech as speechsdk
from datetime import datetime
import openai
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import warnings
import re
import dateparser


warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Global variables
'''processing the audio chunks in a queue'''
audio_queue = queue.Queue()  
'''keeping track of the transcripts in a list for further actions'''
transcripts = []  
last_transcribed_text = ""  # Variable to track the last transcribed text


#This function is essential for initializing the Azure Speech SDK with the correct API keys and settings before you start transcription.
def setup_speech_config():
    languages = [
        "en-US", "fr-FR", "es-ES", "de-DE", "hi-IN",
        "zh-CN", "ja-JP", "ru-RU", "it-IT", "ar-EG"
    ]
    
    speech_key = os.getenv('AZURE_SPEECH_API_KEY')
    speech_region = os.getenv('AZURE_SPEECH_REGION')

    #Checks if both the API key and region are set. If not, it returns None and prompts the user to set those values.

    if not speech_key or not speech_region:
        print("Please set the environment variables SPEECH_KEY and SPEECH_REGION.")
        return None
    
    '''Configures the speechsdk.SpeechConfig object, which includes setting the default language 
    and enabling intermediate results for speaker diarization (i.e., tracking who is speaking)'''

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = languages[0]  # Initially set to english language
    speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults, value='true')
    return speech_config

# Callbacks for transcription events
def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
    print('Canceled event')

def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStopped event')

#A callback function that handles the event when Azure successfully transcribes speech to text.
def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    global last_transcribed_text  # Use the global variable

    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        pc_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        transcripts.append((evt.result.text, pc_time, evt.result.speaker_id))
        
        if evt.result.text != last_transcribed_text:
            print(f'[{pc_time}] Speaker ID({evt.result.speaker_id}): {evt.result.text}')
            last_transcribed_text = evt.result.text
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        pass  # No action needed

def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStarted event')

'''Performs speaker diarization (i.e., figuring out who is speaking when) on an audio chunk using the pyannote library.
This function helps in identifying which speaker is talking at different times in the audio, allowing for better conversation tracking.'''

'''def diarize_audio_with_pyannote(audio_chunk):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=True)
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
    
    return speaker_info'''

# This function prepares a human-readable version of the conversation with speaker labels, making it easier to understand the flow of the conversation.
def format_output(transcripts, speaker_info):
    output = []
    for transcript, timestamp,in transcripts:
        for info in speaker_info:
            if timestamp >= info["start"] and timestamp <= info["end"]:
                output.append(f"Speaker {info['speaker']:02}: {transcript}")
    return "\n".join(output)

# Function to summarize transcriptions using OpenAI API 
def summarize_transcriptions(transcripts):
    combined_transcript = "\n".join([f"Speaker {speaker_id}: {text}" for text, _, speaker_id in transcripts])
    
    # Explicitly inform the model if it's a monologue or dialogue
    if len(set([speaker_id for _, _, speaker_id in transcripts])) == 1:

        combined_transcript = "This is a monologue. Here is the transcript:\n" + combined_transcript
    else:
       
        combined_transcript = "This is a conversation. Here is the transcript:\n" + combined_transcript

    openai.api_key = os.getenv('AI_API_KEY')
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "user", "content": f"Please summarize the following conversation:\n\n{combined_transcript}"}
            ]
        )
        summary = response['choices'][0]['message']['content']
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None

# Function to find dates in the text
def find_dates_in_text(text):
    # This regex handles multiple formats, including "21st May 2003" and "first September in the year of 1995"
    date_regex = r'\b(?:\d{1,2}(?:st|nd|rd|th)? [A-Za-z]+ \d{4}|\d{1,2} [A-Za-z]+ \d{4}|\b[A-Za-z]+ \d{1,2},? \d{4}|first? of [A-Za-z]+ in the year of \d{4})\b | \b\d{2}/\d{2}/\d{4}\b'


    matches = re.findall(date_regex, text)
    found_dates = []

    for match in matches:
        if isinstance(match, str):  # Ensure match is a string
            # Remove ordinal suffix (st, nd, rd, th) if present, to aid date parsing
            match = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', match)
            # Replace "in the year of" to create a standard format
            match = match.replace("in the year of", "").strip()
            # Parse the cleaned date string
            parsed_date = dateparser.parse(match)
            if parsed_date:
                found_dates.append(parsed_date)
    
    return found_dates



# Function to find action items in the text

def find_action_items(text):
    # Regex to match imperative verbs or common action phrases and stop at sentence-ending punctuation
    action_item_regex = r'\b(?:do|follow|submit|review|check|ensure|make sure to|you must|please)\b.*?[.!?]'
    
    matches = re.findall(action_item_regex, text, re.IGNORECASE)  # Find all action-related sentences
    return matches



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
    conversation_transcriber.session_started.connect(conversation_transcriber_session_started_cb)
    conversation_transcriber.session_stopped.connect(conversation_transcriber_session_stopped_cb)
    conversation_transcriber.canceled.connect(conversation_transcriber_recognition_canceled_cb)

    # Stop transcribing on either session stopped or canceled events
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    # Start the transcription
    conversation_transcriber.start_transcribing_async()

    print("Recording... Press Enter to stop.")

    # Wait for user input to stop
    input("Press Enter to stop...")  # Replace keyboard wait with input

    # Stop the Azure Speech SDK transcription
    conversation_transcriber.stop_transcribing_async().get()  # Wait for the stop to complete

    full_transcript_text = " ".join([t[0] for t in transcripts])  # Combine all transcribed text
    dates_found = find_dates_in_text(full_transcript_text)
    if dates_found:
        print("\nDates found in conversation:")
        for date in dates_found:
            print(date.strftime("%Y-%m-%d"))

    action_items_found = find_action_items(full_transcript_text)
    print("Action items found:", action_items_found)

    
    summary = summarize_transcriptions(transcripts)
    if summary:
        print("\nSummary of the conversation:")
        print(summary)


if __name__ == "__main__":
    main()
