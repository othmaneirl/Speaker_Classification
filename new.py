import os
import tkinter as tk
from tkinter import messagebox, ttk
import sounddevice as sd
import librosa
import numpy as np
import torch
import pickle
import matplotlib
import threading
import time
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

try:
    from speechbrain.pretrained import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb"
    )
except Exception as e:
    messagebox.showerror("Error", f"Failed to load SpeechBrain encoder: {e}")
    raise e

try:
    with open("ref_embeddings.pkl", "rb") as f:
        # Expected format: dictionary mapping speaker names (lowercase) to x-vector embeddings (numpy arrays)
        reference_embeddings = pickle.load(f)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load reference embeddings: {e}")
    raise e

# Define a minimal similarity threshold; if best match < threshold then label as "other"
SIMILARITY_THRESHOLD = 0.70  

# -----------------------------
# Global variables for recording & animation
# -----------------------------
recording = False       # Recording state indicator
fs = 16000              # Sample rate in Hz
recorded_frames = []    # List to hold audio data chunks
stream = None           # sounddevice InputStream

# Variables for animation
animation_running = False   # Tracks whether the shuffling animation is active
animation_index = 0         # Tracks which image from the animation list to show next
animation_images = []       # List to hold preloaded PhotoImage objects

# -----------------------------
# Audio callback (invoked by sounddevice)
# -----------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Recording status:", status)
    recorded_frames.append(indata.copy())

# -----------------------------
# Toggle recording state
# -----------------------------
def toggle_recording():
    global recording, stream, recorded_frames
    if not recording:
        # Start recording: clear previous frames and open stream
        recorded_frames = []
        try:
            stream = sd.InputStream(samplerate=fs, channels=1, callback=audio_callback)
            stream.start()
        except Exception as e:
            messagebox.showerror("Error", f"Could not start recording: {e}")
            return
        record_button.config(text="Stop Recording")
        status_label.config(text="Recording...")
        recording = True
        update_plot()  # start updating the waveform plot
    else:
        # Stop recording: close the input stream
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            messagebox.showerror("Error", f"Error stopping recording: {e}")
            return
        recording = False
        record_button.config(text="Start Recording")
        status_label.config(text="Processing audio...")
        # Process audio in a new thread so that the animation runs smoothly.
        threading.Thread(target=process_audio, daemon=True).start()

# -----------------------------
# Update the waveform plot with the cumulative audio data
# -----------------------------
def update_plot():
    if recording:
        if recorded_frames:
            full_data = np.concatenate(recorded_frames, axis=0).flatten()
            x = np.arange(len(full_data))
            line.set_data(x, full_data)
            ax.set_xlim(0, len(full_data))
            if full_data.size > 0:
                y_min, y_max = full_data.min(), full_data.max()
            else:
                y_min, y_max = -1, 1
            ax.set_ylim(y_min - 0.2 * abs(y_min), y_max + 0.2 * abs(y_max))
            canvas.draw()
        app.after(100, update_plot)

# -----------------------------
# Animate shuffling of images (playing card style)
# -----------------------------
def animate_shuffling():
    global animation_index, animation_running
    if animation_running:
        # Cycle through the preloaded images
        current_image = animation_images[animation_index % len(animation_images)]
        picture_label.config(image=current_image)
        picture_label.image = current_image  # keep a reference
        animation_index += 1
        # Change image every 100 ms (change interval to adjust speed)
        app.after(100, animate_shuffling)

# -----------------------------
# Utility function: cosine similarity between two vectors
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# -----------------------------
# Process audio: extract x-vector, predict speaker, and update image
# -----------------------------
def process_audio():
    global animation_running, animation_index
    try:
        # Start the shuffling animation on the main thread
        animation_index = 0
        animation_running = True
        app.after(0, animate_shuffling)
        
        # Record the start time for the animation
        start_time = time.time()

        # Verify that audio data was captured
        if not recorded_frames:
            app.after(0, lambda: status_label.config(text="No audio captured. Please try again."))
            animation_running = False
            return

        # Concatenate recorded frames into one array
        audio_data = np.concatenate(recorded_frames, axis=0).flatten()
        if audio_data.size == 0:
            app.after(0, lambda: status_label.config(text="Empty audio data. Please try again."))
            animation_running = False
            return

        # The SpeechBrain encoder expects a Torch tensor.
        # (Assuming the recorded audio is already at fs=16000)
        audio_tensor = torch.tensor(audio_data).unsqueeze(0)  # shape: [1, num_samples]

        # Extract the x-vector embedding using SpeechBrain (no gradient needed)
        with torch.no_grad():
            embedding = classifier.encode_batch(audio_tensor)
        x_vector = embedding.squeeze(0).detach().cpu().numpy()

        # Compare the x-vector with each reference embedding using cosine similarity
        scores = {}
        for spk, ref_emb in reference_embeddings.items():
            scores[spk] = cosine_similarity(x_vector, ref_emb)
        # Get the speaker with the maximum similarity score
        predicted_speaker = max(scores, key=scores.get)
        max_similarity = scores[predicted_speaker]
        # If the top score is below the threshold, label as "other"
        if max_similarity < SIMILARITY_THRESHOLD:
            predicted_speaker = "other"

        # Determine image file based on the predicted speaker:
        # This example expects that for one of our known speakers we have .jpg images,
        # otherwise, we use "other.png". Adjust this section as needed.
        if predicted_speaker == 0:
            final_filename = "guillaume.jpg"
        elif predicted_speaker == 1:
            final_filename = "othmane.jpg"
        elif predicted_speaker == 2:
            final_filename = "theo.jpg"   
        else:
            final_filename = "other.png"

        # Load and resize the final image using Pillow (size must match animation images)
        try:
            img = Image.open(final_filename)
            img = img.resize((200, 200), Image.LANCZOS)
            final_img = ImageTk.PhotoImage(img)
        except Exception as e_img:
            app.after(0, lambda: status_label.config(text=f"Could not load image: {final_filename}"))
            print(f"Error loading image: {e_img}")
            animation_running = False
            return

        # Define a function to update the UI after the animation delay:
        def update_final_UI():
            global animation_running
            animation_running = False
            picture_label.config(image=final_img)
            picture_label.image = final_img  # keep a reference
            status_label.config(text=f"Predicted: {predicted_speaker}")
        
        # Calculate time elapsed since starting the animation and enforce a minimum duration of 2 seconds
        elapsed_time = time.time() - start_time
        delay_ms = max(0, int((2 - elapsed_time) * 1000))
        app.after(delay_ms, update_final_UI)

    except Exception as e:
        app.after(0, lambda: status_label.config(text="Error during processing"))
        messagebox.showerror("Error", f"An error occurred while processing the audio:\n{e}")
        animation_running = False

# -----------------------------
# Build the GUI
# -----------------------------
app = tk.Tk()
app.title("Speaker Recognition App")
app.geometry("750x600")
app.configure(bg="#f8f9fa")

# Main frame with padding
main_frame = ttk.Frame(app, padding=15)
main_frame.pack(fill=tk.BOTH, expand=True)

# Title label
title_label = ttk.Label(main_frame, text="Speaker Recognition", font=("Helvetica", 22, "bold"))
title_label.pack(pady=10)

# Control frame for recording button and status
control_frame = ttk.Frame(main_frame)
control_frame.pack(pady=10)

record_button = ttk.Button(control_frame, text="Start Recording", command=toggle_recording)
record_button.grid(row=0, column=0, padx=15, pady=10)

status_label = ttk.Label(control_frame, text="Idle", font=("Helvetica", 14))
status_label.grid(row=0, column=1, padx=15, pady=10)

# Frame for prediction (to display animated image and then final image)
prediction_frame = ttk.LabelFrame(main_frame, text="Prediction", padding=10)
prediction_frame.pack(pady=10)

picture_label = ttk.Label(prediction_frame)
picture_label.pack()

# Visualization frame for the recorded audio waveform
viz_frame = ttk.LabelFrame(main_frame, text="Recording Waveform", padding=10)
viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

fig, ax = plt.subplots(figsize=(6, 3))
line, = ax.plot([], [], lw=2, color="#007acc")
ax.set_xlabel("Samples", fontsize=10)
ax.set_ylabel("Amplitude", fontsize=10)
ax.set_title("Live Audio Waveform", fontsize=12)
ax.grid(True)
plt.tight_layout()

canvas = FigureCanvasTkAgg(fig, master=viz_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

# -----------------------------
# Preload images for animation (playing card "shuffling")
# -----------------------------
def preload_animation_images():
    global animation_images
    filenames = ["othmane.jpg", "theo.jpg", "guillaume.jpg", "other.png"]
    for fname in filenames:
        try:
            img = Image.open(fname)
            img = img.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            animation_images.append(photo)
        except Exception as e:
            print(f"Error loading {fname}: {e}")

preload_animation_images()

# Run the GUI event loop
app.mainloop()
