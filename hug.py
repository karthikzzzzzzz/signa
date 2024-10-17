import os
import queue
import azure.cognitiveservices.speech as speechsdk
from datetime import datetime
import openai
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import warnings
import wave
import re
import dateparser

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Global variables
audio_queue = queue.Queue()  
transcripts = []  

class ConversationTranscriber:
    def __init__(self):
        self.speech_config = self.setup_speech_config()
        self.last_transcribed_text = ""

    def setup_speech_config(self):
        speech_key = os.getenv('AZURE_SPEECH_API_KEY')
        speech_region = os.getenv('AZURE_SPEECH_REGION')

        if not speech_key or not speech_region:
            raise ValueError("Please set the environment variables AZURE_SPEECH_API_KEY and AZURE_SPEECH_REGION.")

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_recognition_language = "en-US"  # Set default language
        speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults, value='true')
        return speech_config

    def conversation_transcriber_transcribed_cb(self, evt: speechsdk.SpeechRecognitionEventArgs):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            pc_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            transcripts.append((evt.result.text, pc_time, evt.result.speaker_id))

            if evt.result.text != self.last_transcribed_text:
                print(f'[{pc_time}] Speaker ID({evt.result.speaker_id}): {evt.result.text}')
                self.last_transcribed_text = evt.result.text

    def start_transcribing(self):
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=self.speech_config, audio_config=audio_config)

        # Connect callbacks to events
        conversation_transcriber.transcribed.connect(self.conversation_transcriber_transcribed_cb)
        conversation_transcriber.session_stopped.connect(lambda evt: print('Session stopped.'))
        conversation_transcriber.canceled.connect(lambda evt: print('Transcription canceled.'))

        # Start transcription
        conversation_transcriber.start_transcribing_async()
        print("Recording... Press Enter to stop.")
        input("Press Enter to stop...")
        conversation_transcriber.stop_transcribing_async().get()  # Wait for stop to complete

def format_output(transcripts):
    output = []
    for transcript, timestamp, speaker_id in transcripts:
        output.append(f"Speaker {speaker_id}: {transcript}")
    return "\n".join(output)

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

def find_dates_in_text(text):
    date_regex = r'\b(?:\d{1,2}(?:st|nd|rd|th)? [A-Za-z]+ \d{4}|\d{1,2} [A-Za-z]+ \d{4}|[A-Za-z]+ \d{1,2},? \d{4}|first? of [A-Za-z]+ in the year of \d{4}|\b\d{2}/\d{2}/\d{4}\b)'
    matches = re.findall(date_regex, text)
    found_dates = []

    for match in matches:
        match = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', match)  # Remove ordinal suffix
        match = match.replace("in the year of", "").strip()  # Clean format
        parsed_date = dateparser.parse(match)
        if parsed_date:
            found_dates.append(parsed_date)
    
    return found_dates

def find_action_items(text):
    action_item_regex = r'\b(?:do|follow|submit|review|check|ensure|make sure to|you must|please)\b.*?[.!?]'
    return re.findall(action_item_regex, text, re.IGNORECASE)

def main():
    try:
        transcriber = ConversationTranscriber()
        transcriber.start_transcribing()

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
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
