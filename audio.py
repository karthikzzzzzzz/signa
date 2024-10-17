import pyaudio
import wave
import threading

def record_audio_until_keypress(filename):
    """Records audio from the microphone until Enter is pressed and saves it to a file.

    Args:
        filename (str): The name of the file to save the recording.
    """
    # Set the parameters for recording
    FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
    CHANNELS = 1              # Number of channels (1 for mono)
    RATE = 44100              # Sample rate (samples per second)
    CHUNK = 1024              # Buffer size (number of frames per buffer)

    # Create a PyAudio object
    p = pyaudio.PyAudio()

    # Open a stream for recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording... Press Enter to stop.")

    frames = []

    # Function to record audio
    def record():
        while not stop_event.is_set():
            data = stream.read(CHUNK)
            frames.append(data)

    # Create a thread to record audio in the background
    stop_event = threading.Event()
    record_thread = threading.Thread(target=record)
    record_thread.start()

    # Wait for Enter to be pressed
    input()
    stop_event.set()
    record_thread.join()

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# Usage example
if __name__ == "__main__":
    record_audio_until_keypress("out.wav")  # Record audio until Enter is pressed.
