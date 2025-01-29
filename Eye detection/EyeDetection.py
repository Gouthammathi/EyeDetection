import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from webcolors import rgb_to_name

# Function to detect eyes in an image and extract basic features
def detect_and_analyze_eyes(frame):
    global max_display_width, max_display_height
    
    # Calculate the scaling factors for resizing the image to fit within the window
    width_ratio = max_display_width / frame.shape[1]
    height_ratio = max_display_height / frame.shape[0]
    min_ratio = min(width_ratio, height_ratio)
    
    # Resize the frame while maintaining its aspect ratio
    frame = cv2.resize(frame, None, fx=min_ratio, fy=min_ratio)
    
    # Convert the frame to grayscale for eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load the haarcascade for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Convert the frame to RGB format for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create a PIL ImageTk for displaying in the GUI
    image = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image)
    
    # Display the frame with detected eyes on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.photo = photo
    
    # Basic feature extraction
    image_size = (frame.shape[1], frame.shape[0])  # Width x Height
    average_color_bgr = np.mean(frame, axis=(0, 1))  # Average color in BGR format
    average_color_rgb = (int(average_color_bgr[2]), int(average_color_bgr[1]), int(average_color_bgr[0]))  # Convert to RGB as integers

    # Convert RGB color to a human-readable color name using webcolors
    try:
        average_color_name = rgb_to_name(average_color_rgb)
    except ValueError:
        average_color_name = "Unknown"
    
    # Create a label for displaying features
    feature_text = f"Image Size: {image_size[0]} x {image_size[1]}\n"
    feature_text += f"Average Color (RGB): {average_color_rgb}\n"
    feature_text += f"Average Color Name: {average_color_name}\n"
    feature_text += f"Number of Detected Eyes: {len(eyes)}"
    feature_label.config(text=feature_text)

# Create a VideoCapture object to capture video from your camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Create a GUI window
window = tk.Tk()
window.title("Eye Detection and Feature Extraction")

# Set the maximum display width and height
max_display_width = 640
max_display_height = 480

# Create a canvas for displaying the camera feed
canvas = tk.Canvas(window, width=max_display_width, height=max_display_height)
canvas.pack()

# Create a label for displaying features
feature_label = tk.Label(window, text="", padx=10, pady=10)
feature_label.pack()

def update_frame():
    ret, frame = cap.read()
    if ret:
        detect_and_analyze_eyes(frame)
    window.after(10, update_frame)

# Call the update_frame function to continuously update the camera feed
update_frame()

# Start the GUI application
window.mainloop()

# Release the camera when finished
cap.release()
