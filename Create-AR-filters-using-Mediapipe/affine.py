import cv2
import numpy as np
import imageio

def draw_triangle(image, points, color):
    points = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

def create_translation_gif(triangle_points, steps=20):
    frames = []
    for i in range(steps):
        translation_matrix = np.array([[1, 0, 50 * i / steps],
                                       [0, 1, 50 * i / steps]], dtype=np.float32)
        transformed_points = cv2.transform(np.array([triangle_points]), translation_matrix)[0]
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        draw_triangle(frame, transformed_points, (255, 255, 255))
        frames.append(frame)
    imageio.mimsave('translation.gif', frames, fps=10, loop=0)

def create_scaling_gif(triangle_points, steps=20):
    frames = []
    for i in range(steps):
        scale_factor = 1 + 0.5 * i / steps
        scaling_matrix = np.array([[scale_factor, 0, 0],
                                   [0, scale_factor, 0]], dtype=np.float32)
        transformed_points = cv2.transform(np.array([triangle_points]), scaling_matrix)[0]
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        draw_triangle(frame, transformed_points, (255, 255, 255))
        frames.append(frame)
    imageio.mimsave('scaling.gif', frames, fps=10, loop=0)

def create_rotation_gif(triangle_points, steps=20):
    frames = []
    center = (200, 200)
    for i in range(steps):
        angle = 45 * i / steps
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        transformed_points = cv2.transform(np.array([triangle_points]),rotation_matrix)[0]
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        draw_triangle(frame, transformed_points, (255, 255, 255))
        frames.append(frame)
    imageio.mimsave('rotation.gif', frames, fps=10, loop=0)

def create_shearing_gif(triangle_points, steps=20):
    frames = []
    for i in range(steps):
        shear_factor = 0.5 * i / steps
        shearing_matrix = np.array([[1, shear_factor, 0],
                                    [shear_factor, 1, 0]], dtype=np.float32)
        transformed_points = cv2.transform(np.array([triangle_points]), shearing_matrix)[0]
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        draw_triangle(frame, transformed_points, (255, 255, 255))
        frames.append(frame)
    imageio.mimsave('shearing.gif', frames, fps=10, loop=0)

def create_reflection_gif(triangle_points, steps=20):
    frames = []
    for i in range(steps):
        reflection_factor = 1 - 2 * (i / steps)
        reflection_matrix = np.array([[reflection_factor, 0, 400 * (1 - reflection_factor) / 2],
                                      [0, reflection_factor, 400 * (1 - reflection_factor) / 2]], dtype=np.float32)
        transformed_points = cv2.transform(np.array([triangle_points]), reflection_matrix)[0]
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        draw_triangle(frame, transformed_points, (255, 255, 255))
        frames.append(frame)
    imageio.mimsave('reflection.gif', frames, fps=10, loop=0)

# Define the original triangle points
triangle_points = np.array([[100, 300], [200, 100], [300, 300]], dtype=np.float32)

# Create GIFs for each transformation
create_translation_gif(triangle_points)
create_scaling_gif(triangle_points)
create_rotation_gif(triangle_points)
create_shearing_gif(triangle_points)
create_reflection_gif(triangle_points)

print("GIFs created for each transformation.")
