import pyaudio
import wave
import os
import whisper
import warnings
# from transformers import BartForConditionalGeneration, BartTokenizer
"UNLABELLED SPEAKERS"
# Ignore all warnings
warnings.filterwarnings("ignore")

def record_chunk(p, stream, file_path, chunk_length=5):
    
    
    """Records an audio chunk and saves it to a file."""
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        try:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        except OSError as e:
            print(f"Error while recording: {e}")
    
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_chunk(model, chunk_file):
    """Transcribes the recorded chunk using the Whisper model."""
    result = model.transcribe(chunk_file)
    return result["text"]
'''def summarize(text, model, tokenizer):
    """Summarizes the input text using the BART model."""
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary'''

def main():
    model_size = "small.en"  # Use a smaller model for better performance if needed
    model = whisper.load_model(model_size)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    #summarization_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    #tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    accumulated_transcription = ""

    try:
        while True:
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)
            transcription = transcribe_chunk(model, chunk_file)
            print(transcription)
            os.remove(chunk_file)
            accumulated_transcription += transcription + " "
    except KeyboardInterrupt:
        print("Stopping...")
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
    finally:
        print("Final transcription: " + accumulated_transcription)
        #summary = summarize(accumulated_transcription, summarization_model, tokenizer)
        #print("Summary: " + summary)
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
