import streamlit as st
import subprocess
import soundfile
import os
import time
import tempfile
from huggingface_hub import login
import random
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from IPython.display import Audio
from st_audiorec import st_audiorec

login(token="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")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)


prompt_templates = {
    "happy": [
        "A joyful [genre] piece with an upbeat [instrument] featuring a lively, cheerful melody, capturing the feeling of happiness and celebration.",
        "An energetic [genre] track with a bright, lively [instrument], evoking a sense of joy and excitement.",
        "A cheerful [genre] tune with a playful [instrument], radiating positive energy and fun.",
        "A vibrant [genre] melody with an upbeat [instrument], creating a sense of joy and contentment.",
        "A spirited [genre] composition with a happy [instrument], bringing out the feeling of elation and fun.",
        "A lively [genre] piece with an uplifting [instrument], expressing joy and cheerfulness.",
        "An optimistic [genre] track with a peppy [instrument], evoking feelings of happiness and positivity."
    ],
    "sad": [
        "A sad [genre] piece with a [instrument] featuring a slow, haunting melody, capturing the feeling of deep sorrow and reflection.",
        "A melancholic [genre] track with a gentle, sorrowful [instrument], conveying a sense of loss and introspection.",
        "A somber [genre] melody with a mournful [instrument], expressing a deep sense of sadness and melancholy.",
        "A wistful [genre] composition with a soft, sorrowful [instrument], evoking feelings of longing and heartache.",
        "A reflective [genre] piece with a poignant [instrument], capturing the essence of sadness and introspection.",
        "A mournful [genre] track with a touching [instrument], depicting sorrow and grief.",
        "A poignant [genre] melody with a gentle [instrument], expressing a sense of sadness and longing."
    ],
    "neutral": [
        "A neutral [genre] piece with a balanced [instrument], providing a calm and steady melody.",
        "An even-tempered [genre] track with a smooth [instrument], evoking a sense of calm and relaxation.",
        "A serene [genre] melody with a gentle [instrument], creating a tranquil and peaceful atmosphere.",
        "A calm [genre] composition with a soothing [instrument], promoting a sense of balance and neutrality.",
        "A mellow [genre] tune with a soft [instrument], offering a sense of steadiness and calm.",
        "A tranquil [genre] piece with a gentle [instrument], fostering a peaceful and relaxed mood.",
        "A balanced [genre] track with a harmonious [instrument], inducing a sense of calm and serenity."
    ],
    "angry": [
        "An intense [genre] piece with an aggressive [instrument], featuring a fast, powerful melody that captures the feeling of anger.",
        "A furious [genre] track with a heavy, driving [instrument], conveying a sense of rage and intensity.",
        "A fierce [genre] melody with a strong, relentless [instrument], expressing anger and aggression.",
        "An aggressive [genre] composition with a hard-hitting [instrument], reflecting the intensity of anger.",
        "A powerful [genre] piece with a fast, forceful [instrument], embodying the essence of fury and wrath.",
        "A stormy [genre] track with a harsh [instrument], depicting feelings of anger and frustration.",
        "An explosive [genre] melody with a vigorous [instrument], expressing intense anger and fury."
    ],
    "surprise": [
        "An unexpected [genre] piece with a dynamic [instrument], featuring a lively, surprising melody that captures the feeling of surprise.",
        "A spontaneous [genre] track with an exciting [instrument], evoking a sense of wonder and amazement.",
        "An unpredictable [genre] melody with a vibrant [instrument], creating a sense of surprise and excitement.",
        "An astonishing [genre] composition with a lively [instrument], reflecting the feeling of surprise and wonder.",
        "A surprising [genre] piece with a dynamic [instrument], capturing the essence of unpredictability and amazement.",
        "A whimsical [genre] track with a playful [instrument], evoking a sense of unexpected delight.",
        "An adventurous [genre] melody with an energetic [instrument], expressing surprise and excitement."
    ]
}
genres_instruments = {
    "happy": {
        "genres": ["pop", "dance", "reggae", "ska", "funk", "disco", "soul", "rock", "latin", "country"],
        "instruments": ["guitar", "ukulele", "saxophone", "violin", "piano", "trumpet", "drums", "bass", "harmonica", "flute"]
    },
    "sad": {
        "genres": ["classical", "piano", "blues", "folk", "ambient", "acoustic", "singer-songwriter", "jazz ballad", "gospel", "soft rock"],
        "instruments": ["piano", "violin", "cello", "acoustic guitar", "flute", "clarinet", "harp", "saxophone", "oboe", "bass"]
    },
    "neutral": {
        "genres": ["ambient", "chillout", "instrumental", "new age", "minimalism", "easy listening", "lounge", "world music", "electronic", "downtempo"],
        "instruments": ["synth", "flute", "harp", "chimes", "guitar", "piano", "violin", "cello", "marimba", "handpan"]
    },
    "angry": {
        "genres": ["rock", "metal", "punk", "hardcore", "industrial", "grunge", "rap", "hip-hop", "drum and bass", "EDM"],
        "instruments": ["electric guitar", "drums", "bass", "synth", "trumpet", "trombone", "saxophone", "vocals", "turntables", "keyboard"]
    },
    "surprise": {
        "genres": ["jazz", "fusion", "experimental", "avant-garde", "progressive rock", "world music", "psychedelic", "electronic", "trap", "techno"],
        "instruments": ["trumpet", "saxophone", "flute", "vibraphone", "synth", "electric guitar", "drums", "bass", "keyboard", "accordion"]
    }
}

def generate_prompt(emotion):
    if emotion in prompt_templates and emotion in genres_instruments:
        template = random.choice(prompt_templates[emotion])
        genre = random.choice(genres_instruments[emotion]['genres'])
        instrument = random.choice(genres_instruments[emotion]['instruments'])
        return template.replace("[genre]", genre).replace("[instrument]", instrument)
    else:
        return 'Emotion not recognized.'

def main():
    st.markdown("""
        <h1 style='text-align: center; font-size: 3em;'>
            🎵 EmoFlow 🎵
        </h1>
    """, unsafe_allow_html=True)
    st.header("Mood-Based Music Generation System")

    # Sidebar for user input
    st.sidebar.title("User Input")
    user_choice = st.sidebar.selectbox("Choose an option", ["Facial Emotion Detection", "Speech Emotion Recognition"])

    # Placeholder for results
    st.write("## Results")

    if user_choice == "Facial Emotion Detection":
        st.write("Facial Emotion Detection functionality here")
        # Placeholder for file uploader
        user_select = st.selectbox("Choose an option", ["Upload an Image", "Take a Photo"])

        if user_select == "Upload an Image":
          
          uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
          if uploaded_file is not None:
              # Process the uploaded image file here
              st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
              st.write("Detecting emotion...")

              # Save the uploaded file temporarily
              with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                  temp_file.write(uploaded_file.getvalue())
                  temp_file_path = temp_file.name

                      
              # Perform emotion detection
              # Define the command to run
              command = [
                  "python", './yolov9/detect.py',
                  "--img", "640",
                  "--conf", "0.1",
                  "--device", "0",
                  "--weights", './best.pt',
                  "--source", temp_file_path
              ]

                      # Run the command and capture the output
              result = subprocess.run(command, capture_output=True, text=True)

              # Get the standard output
              detected_emotion = result.stdout
              st.write(f"You look like {detected_emotion} today!")
              # Display the detected emotion and recommended music

        elif user_select == "Take a Photo":
            # Placeholder for file uploader
            img_file_buffer = st.camera_input("Take a picture")

            if img_file_buffer is not None:
                # To read image file buffer as bytes:
                bytes_data = img_file_buffer.getvalue()
                # Check the type of bytes_data:
                # Should output: <class 'bytes'>
                #st.write(type(bytes_data))
            # # Placeholder for camera input
            # camera = st.camera_input("Take a photo")
            # if camera is not None:
            #     # Process the camera input here
                st.image(img_file_buffer, caption="Camera Input", use_column_width=True)
                st.write("Detecting emotion...")

                # Save the camera input temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(bytes_data)
                    temp_file_path = temp_file.name

                
                # Perform emotion detection
                # Define the command to run
                command = [
                    "python", './yolov9/detect.py',
                    "--img", "640",
                    "--conf", "0.1",
                    "--device", "0",
                    "--weights", './best.pt',
                    "--source", temp_file_path
                ]

                 # Run the command and capture the output
                result = subprocess.run(command, capture_output=True, text=True)

                # Get the standard output
                detected_emotion = result.stdout
                st.write(f"You look like {detected_emotion} today!")
                # Display the detected emotion and recommended music
                
            


    elif user_choice == "Speech Emotion Recognition":
        st.write("Speech Emotion Recognition functionality here")
        # Placeholder for file uploader
        user_select = st.selectbox("Choose an option", ["Upload an audio file", "Record an Audio"])

        if user_select == "Upload an audio file":



          uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
          if uploaded_audio is not None:
              # Process the uploaded audio file here
              st.audio(uploaded_audio, format="audio/wav")
              st.write("Detecting emotion...")

              # Save the uploaded file temporarily
              with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                  temp_file.write(uploaded_audio.getvalue())
                  temp_file_path = temp_file.name

              
              
              # Define the command to run
              command = [
                  "python", './model/speech.py',
                  "--path", temp_file_path
              ]

              # Run the command and capture the output
              result = subprocess.run(command, capture_output=True, text=True)
              
              detected_emotion = result.stdout
              detected_emotion_1 = detected_emotion.splitlines()[-1]
              
              st.write(f"You look like {detected_emotion_1} today!")
              # Display the detected emotion and recommended music
          
        elif user_select == "Record an Audio":
          audio_bytes = st_audiorec()
          if audio_bytes is not None:
              #st.audio(audio_bytes, format="audio/wav")
              st.write("Detecting emotion...")

                # Save the uploaded file temporarily
              with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                  temp_file.write(audio_bytes)
                  temp_file_path = temp_file.name

              
              
              # Define the command to run
              command = [
                  "python", './model/speech.py',
                  "--path", temp_file_path
              ]

              # Run the command and capture the output
              result = subprocess.run(command, capture_output=True, text=True)
              
              detected_emotion = result.stdout
              detected_emotion_1 = detected_emotion.splitlines()[-1]
              
              st.write(f"You look like {detected_emotion_1} today!")
              # Display the detected emotion and recommended music



    # Add a button for generating music
    if st.button("Generate Music"):
        st.write("Generating music based on detected emotion...")
        if detected_emotion is not None:
          if user_choice == "Facial Emotion Detection":
            # Generate music based on detected emotion
            prompt = generate_prompt(detected_emotion.split('\n')[0])
            st.write(f"Music Prompt: {prompt}")
          
          elif user_choice == "Speech Emotion Recognition":
            # Generate music based on detected emotion
            prompt = generate_prompt(detected_emotion.splitlines()[-1])
            st.write(f"Music Prompt: {prompt}")
          else:
            st.write("Prompt not generated.")

        # Set up text and timing conditioning
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": 30
        }]

        # Generate stereo audio
        output = generate_diffusion_cond(
            model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        audio_path = "output.wav"
        torchaudio.save(audio_path, output, sample_rate)
        st.audio(audio_path)

if __name__ == "__main__":
    main()
