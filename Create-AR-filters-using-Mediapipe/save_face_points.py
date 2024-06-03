import mediapipe as mp
import cv2
import numpy as np
import imageio

VISUALIZE_FACE_POINTS = True

# detect facial landmarks in image
def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]

    height, width = img.shape[:-1]

    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0

        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))

            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y

            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []

            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0

# process input from video file
cap = cv2.VideoCapture('output.avi')

output_frames = []

# The main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        points2 = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # if face is partially detected
        if not points2 or (len(points2) != 75):
            continue

        # Visualize face points
        if VISUALIZE_FACE_POINTS:
            for idx, point in enumerate(points2):
                cv2.circle(frame, point, 2, (255, 0, 0), -1)
                cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)

        # Append output frame to list
        output_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # cv2.imshow("Face Points", frame)

        keypressed = cv2.waitKey(1) & 0xFF
        if keypressed == 27:
            break

cap.release()
cv2.destroyAllWindows()

# Save frames as a GIF
imageio.mimsave('output_face_points.gif', output_frames, fps=20, loop=0)

print("GIF saved to 'output_face_points.gif'")
