import cv2
import imageio

# Load the video file
video_path = '../output_hand.avi'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Read frames from the video
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

# Release the video capture object
cap.release()

# Save frames as a GIF with loop set to 0 (infinite loop)
output_gif_path = '../output_hand.gif'
imageio.mimsave(output_gif_path, frames, fps=20, loop=0)

print(f"GIF saved to {output_gif_path}")
