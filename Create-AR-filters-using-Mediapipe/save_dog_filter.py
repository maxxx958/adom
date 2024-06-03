import mediapipe as mp
import cv2
import numpy as np
import faceBlendCommon as fbc
import imageio
import csv

# Filters configuration for non-morphing filters (example with 'dog')
filters_config = {
    'dog': [{'path': "filters/dog-ears.png",
             'anno_path': "filters/dog-ears_annotations.csv",
             'morph': False, 'animated': False, 'has_alpha': True},
            {'path': "filters/dog-nose.png",
             'anno_path': "filters/dog-nose_annotations.csv",
             'morph': False, 'animated': False, 'has_alpha': True}],
}

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
            for idx, value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')
            relevant_keypnts = []
            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0

def load_filter_img(img_path, has_alpha):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))
    else:
        alpha = np.ones(img.shape[:2], dtype=img.dtype) * 255  # Create a full opacity alpha channel

    return img, alpha

def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points

def load_filter(filter_name="dog"):
    filters = filters_config[filter_name]
    multi_filter_runtime = []
    for filter in filters:
        temp_dict = {}
        img1, img1_alpha = load_filter_img(filter['path'], filter['has_alpha'])
        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha
        points = load_landmarks(filter['anno_path'])
        temp_dict['points'] = points
        multi_filter_runtime.append(temp_dict)
    return filters, multi_filter_runtime

cap = cv2.VideoCapture('output.avi')

# Get fps from the input video
fps = cap.get(cv2.CAP_PROP_FPS)

iter_filter_keys = iter(filters_config.keys())
filters, multi_filter_runtime = load_filter("dog")

output_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        points2 = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not points2 or (len(points2) != 75):
            continue

        for idx, filter in enumerate(filters):
            filter_runtime = multi_filter_runtime[idx]
            img1 = filter_runtime['img']
            points1 = filter_runtime['points']
            img1_alpha = filter_runtime['img_a']

            dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
            tform = fbc.similarityTransform(list(points1.values()), dst_points)
            trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
            trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
            mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

            mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
            mask2 = (255.0, 255.0, 255.0) - mask1

            temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
            temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
            output = temp1 + temp2

            frame = output = np.uint8(output)
            output_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cv2.imshow("Face Filter", output)
        keypressed = cv2.waitKey(1) & 0xFF
        if keypressed == 27:
            break

cap.release()
cv2.destroyAllWindows()

# Use the retrieved fps to save the GIF with compression settings
imageio.mimsave('output_dog.gif', output_frames, fps=fps, loop=0, subrectangles=True, quantizer='nq')

print("GIF saved as 'output_dog.gif'")
