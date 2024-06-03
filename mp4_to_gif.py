import moviepy.editor as mp

# Load the video file
video_path = "demo_vid.mp4"
video_clip = mp.VideoFileClip(video_path)

# Define the output path for the gif
gif_path = "demo_vid.gif"

# Convert the video clip to gif
video_clip.write_gif(gif_path, fps=10)

print("GIF saved at:", gif_path)
