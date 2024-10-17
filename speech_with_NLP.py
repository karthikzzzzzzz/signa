'''import speech_recognition as sp
speech_recognition library:- It is a Python package that provides functionalities for recognizing speech in audio files or live 
audio streams. It offers a simple interface to convert spoken language into text, making it easier to integrate voice recognition 
capabilities into applications. 

from transformers import T5Tokenizer, T5ForConditionalGeneration

T5 Tokenizer: the text that is obtained from the audio is converted or mapped into some numerical ids where each id represents some text meaning 
T5model: its a pre-trained model that is used to generate text from the mapped ids and returns the text into human-readable format.

print("Starting...")

# Function to listen and transcribe audio to text
def listen_and_transcript():
    recognizer = sp.Recognizer() #it recognises the audio or the speech/voice
    with sp.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        print("Processing...")

        try:
            text = recognizer.recognize_google(audio)
            print("Transcription:",text)
            return text
        except sp.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sp.RequestError as e:
            print("Could not request results:",e)
            return None

# Function to summarize the transcript
def summarize_text(text):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Tuning the input for summarization
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Adjusting the parameters for better summarization
    summary_ids = model.generate(
        inputs,
        max_length=100,  # Lower max length for a more concise summary
        min_length=25,  # Set a reasonable minimum length for a precise summary
        length_penalty=2.5,  # Increasing the penalty for longer summaries
        num_beams=6,  # Higher number of beams for better output quality
        early_stopping=True
    )
    
    
    # Decoding the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # Write the summary to a file
    with open("summary.txt", "w") as file:
        file.write(summary)

    print("Summary has been written to 'summary.txt'")
    return summary

if __name__ == "__main__":
    transcript_text = listen_and_transcript()
    if transcript_text:
        summarize_text(transcript_text)
'''
import pyaudio
import numpy as np
import whisper
import torch

# Load the Whisper model
model = whisper.load_model("base")

# Setup PyAudio to capture audio from microphone
CHUNK = 1024  # Buffer size
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Mono audio
RATE = 16000  # Sampling rate (Whisper model prefers 16kHz)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open microphone stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording... Press Ctrl+C to stop.")

try:
    while True:
        # Read audio data from microphone
        data = stream.read(CHUNK)
        
        # Convert the audio data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Whisper prefers a 1D tensor of audio at 16000 Hz
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)

        # Perform transcription using Whisper
        transcription = model.transcribe(audio_tensor)

        # Print the transcription in real-time
        print(transcription['text'])

except KeyboardInterrupt:
    # When user stops recording
    print("Recording stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
