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


# Record audio from the user and save it as "sample_audio.wav"
def record_audio(filename, time_recorded_seconds):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
# Recording the user
    print("Recording... Press Ctrl+C to stop recording early.")
    frames = []

    try:
        start_time = time.time()
       
        while time.time() - start_time < time_recorded_seconds:
            
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("\nRecording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Main function 
def main():

    model = whisper.load_model('base')
    

    time_recorded_seconds = 10
    run_loop = True
    while run_loop:
        user_input = input("Press the ENTER to start recording.\nYou will have 10 seconds to talk.")
        if user_input.lower() == '': 
            record_audio("sample_audio.wav", time_recorded_seconds)  # Record until user presses Enter
            run_loop = False

    # Load audio and pad/trim it to 30 seconds of expected input
    audio = whisper.load_audio("sample_audio.wav")
    audio = whisper.pad_or_trim(audio)

    # Making log-Mel spectrogram and moving to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detecting spoken language in input
    _, probs = model.detect_language(mel)
    print(f"Detected Language: {max(probs, key=probs.get)}")

    # Decoding the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # Printing the recognized text
    audio_txt = result.text
    print(audio_txt)

    # Reading API key from the file
    with open('hidden_key.txt', 'r') as key_file:
        api_key = key_file.read().strip()

    # Setting up OpenAI API key
    openai.api_key = api_key

    # Creating conversation with initial system message
    messages = [
        {"role": "system", "content": "You are a kind wizard of vast wisdom"
          "named 'Seer', who summarizes question and statements made from 'audio.txt'"
          "while concisely replying in 2-3 sentences."}, 
    ]

    # Adding user's transcription to the conversation
    messages.append(
        {"role": "user", "content": audio_txt},
    )

    # Sending conversation to OpenAI
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )

    # Receiving and printing assistant's reply
    reply = chat.choices[0].message.content
    print(f"\nSeer: {reply}")

    # Appending assistant's reply to the conversation
    messages.append({"role": "assistant", "content": reply})

    # Setting speech for Seer's reply
    language = "en"
    speech = gTTS(text=reply, lang=language, slow=False, tld="ca")
    speech.save("seer_reply.wav")

    # Loading Seer's reply audio file
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



