import tkinter as tk
from tkinter import messagebox, ttk
import sounddevice as sd
import librosa
import numpy as np
import tensorflow as tf
import pickle
import matplotlib
import threading
import time
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# -----------------------------
# Load model and preprocessing objects
# -----------------------------
try:
    model = tf.keras.models.load_model("model_on.h5")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model.h5: {e}")
    raise e

try:
    with open("scaler-4.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load scaler.pkl: {e}")
    raise e

try:
    with open("encoder-4.pkl", "rb") as f:
        encoder = pickle.load(f)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load encoder.pkl: {e}")
    raise e

# -----------------------------
# Global variables
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
def audio_callback(indata, frames, time, status):
    if status:
        print("Recording status:", status)
    recorded_frames.append(indata.copy())

# -----------------------------
# Toggle recording state
# -----------------------------
def toggle_recording():
    global recording, stream, recorded_frames
    if not recording:
        # Start recording: reset previous data and open stream
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
        # Start processing in a new thread so that the animation runs smoothly.
        threading.Thread(target=process_audio, daemon=True).start()

# -----------------------------
# Update the plot to show the cumulative recording data
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
        # Cycle through animation_images (assumes the list is already preloaded)
        current_image = animation_images[animation_index % len(animation_images)]
        picture_label.config(image=current_image)
        picture_label.image = current_image  # keep a reference
        animation_index += 1
        # Change image every 100 ms; adjust to change animation speed
        app.after(100, animate_shuffling)

# -----------------------------
# Process audio: extract features, predict speaker, and update image
# -----------------------------
def process_audio():
    global animation_running, animation_index
    try:
        # Start the animation on the main thread
        animation_index = 0
        animation_running = True
        app.after(0, animate_shuffling)

        # Verify that audio data was captured
        if not recorded_frames:
            app.after(0, lambda: status_label.config(text="No audio captured. Please try again."))
            animation_running = False
            return

        audio_data = np.concatenate(recorded_frames, axis=0).flatten()
        if audio_data.size == 0:
            app.after(0, lambda: status_label.config(text="Empty audio data. Please try again."))
            animation_running = False
            return

        # Compute MFCC features (13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=fs, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1).reshape(1, -1)
        mfccs_scaled = scaler.transform(mfccs_mean)

        # Simulate a slight delay if needed (e.g., for visual effect)
        # time.sleep(2)  # Uncomment to demonstrate animation longer

        # Predict the speaker with the pre-trained model
        predictions = model.predict(mfccs_scaled)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_speaker = encoder.inverse_transform(predicted_class)[0]

        # Determine image file based on the predicted speaker:

        if predicted_speaker.lower() in ["othmane", "theo", "guillaume"]:
            final_filename = f"{predicted_speaker.lower()}.jpg"
        else:
            final_filename = "other.png"

        # Load and resize the final image using Pillow (use the same size as animation images)
        try:
            img = Image.open(final_filename)
            img = img.resize((200, 200), Image.LANCZOS)
            final_img = ImageTk.PhotoImage(img)
        except Exception as e_img:
            app.after(0, lambda: status_label.config(text=f"Could not load image: {final_filename}"))
            print(f"Error loading image: {e_img}")
            animation_running = False
            return

        # Stop animation and update final image on the main thread
        animation_running = False
        app.after(0, lambda: picture_label.config(image=final_img))
        picture_label.image = final_img  # keep a reference
        app.after(0, lambda: status_label.config(text=f"Predicted: {predicted_speaker}"))
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

# Create the main frame with padding
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

# Frame for prediction output (to display animated image and then final image)
prediction_frame = ttk.LabelFrame(main_frame, text="Prediction", padding=10)
prediction_frame.pack(pady=10)

picture_label = ttk.Label(prediction_frame)
picture_label.pack()

# Visualization frame for the waveform
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
# Preload images for animation
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
