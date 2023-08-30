import whisper
import openai
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import subprocess
import sys
import pyaudio
import wave
import time

# Function to record audio from the user and save it as "sample_audio.wav"
def record_audio(filename, time_recorded_seconds):
    # Configuration for audio recording
    CHUNK = 1024  # Number of audio frames per buffer
    FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
    CHANNELS = 1  # Number of audio channels (mono)
    RATE = 44100  # Sampling rate in Hz

    # Initialize PyAudio instance
    p = pyaudio.PyAudio()

    # Open audio stream for recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording... Press Ctrl+C to stop recording early.")
    frames = []

    try:
        start_time = time.time()
        # Record audio until specified time
        while time.time() - start_time < time_recorded_seconds:
            data = stream.read(CHUNK)  # Read audio data in chunks
            frames.append(data)  # Append data to frames list
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        # Cleanup after recording
        stream.stop_stream()  # Stop audio stream
        stream.close()  # Close audio stream
        p.terminate()  # Terminate PyAudio instance

        # Save recorded frames as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)  # Set number of channels
        wf.setsampwidth(p.get_sample_size(FORMAT))  # Set sample width
        wf.setframerate(RATE)  # Set frame rate
        wf.writeframes(b''.join(frames))  # Write frames to WAV file
        wf.close()  # Close WAV file

# Main function begins
def main():
    # Load Whisper model
    model = whisper.load_model('base')
    
    # Set the duration for recording
    time_recorded_seconds = 10
    run_loop = True
    while run_loop:
        user_input = input("Press ENTER to start recording.\nYou will have 10 seconds to talk.")
        if user_input.lower() == '':
            # Record audio until user presses Enter
            record_audio("sample_audio.wav", time_recorded_seconds)
            run_loop = False

    # Load audio and pad/trim it to 30 seconds of expected input
    audio = whisper.load_audio("sample_audio.wav")
    audio = whisper.pad_or_trim(audio)

    # Generate log-Mel spectrogram and move it to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect spoken language in input
    _, probs = model.detect_language(mel)
    print(f"Detected Language: {max(probs, key=probs.get)}")

    # Decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # Get recognized text
    audio_txt = result.text
    print(audio_txt)

    # Read API key from the file
    with open('hidden_key.txt', 'r') as key_file:
        api_key = key_file.read().strip()

    # Set up OpenAI API key
    openai.api_key = api_key

    # Create conversation with initial system message
    messages = [
        {"role": "system", "content": "You are a kind wizard of vast wisdom"
          "named 'Seer', who summarizes question and statements made from 'audio.txt'"
          "while concisely replying in 2-3 sentences."}, 
    ]

    # Add user's transcription to the conversation
    messages.append(
        {"role": "user", "content": audio_txt},
    )

    # Send conversation to OpenAI
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )

    # Get assistant's reply
    reply = chat.choices[0].message.content
    print(f"\nSeer: {reply}")

    # Append assistant's reply to the conversation
    messages.append({"role": "assistant", "content": reply})

    # Set up speech for Seer's reply
    language = "en"
    speech = gTTS(text=reply, lang=language, slow=False, tld="ca")
    speech.save("seer_reply.wav")

    # Load Seer's reply audio file
    audio = AudioSegment.from_file("seer_reply.wav")
    audio_duration = len(audio) / 1000  # Convert milliseconds to seconds

    # Command to play the audio using FFmpeg, suppressing output
    ffmpeg_command = ["ffplay", "-nodisp", "seer_reply.wav"]

    # Try-catch for execution time 
    try:
        subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=audio_duration)
    except subprocess.TimeoutExpired:
        pass

    # Exit the program
    exit()

if __name__ == "__main__":
    main()
