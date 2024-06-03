import cv2
import numpy as np
import faceBlendCommon as fbc
import csv

# Function to load filter image
def load_filter_img(img_path, has_alpha):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError(f"Error: Unable to read image at path '{img_path}'. Please check the file path and ensure the file exists.")
    
    alpha = None
    if has_alpha:
        if img.shape[2] == 4:
            b, g, r, alpha = cv2.split(img)
            img = cv2.merge((b, g, r))
        else:
            raise ValueError(f"Error: Image at path '{img_path}' does not have an alpha channel.")
    
    return img, alpha

# Function to load landmarks from CSV
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

# Load the filter image and landmarks
filter_img_path = 'filters/anonymous.png'
annotation_file_path = 'filters/anonymous_annotations.csv'
img, alpha = load_filter_img(filter_img_path, has_alpha=True)
points = load_landmarks(annotation_file_path)

def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex

# Find convex hull
hull, hullIndex = find_convex_hull(points)

# Create a copy of the image to draw convex hull
img_hull = img.copy()

# Draw convex hull
for i in range(len(hull)):
    cv2.line(img_hull, hull[i], hull[(i + 1) % len(hull)], (0, 255, 0), 2)

# Save the image with convex hull
cv2.imwrite('convex_hull.png', img_hull)

# Compute Delaunay triangulation
sizeImg1 = img.shape
rect = (0, 0, sizeImg1[1], sizeImg1[0])
dt = fbc.calculateDelaunayTriangles(rect, hull)

# Create a copy of the image to draw Delaunay triangles
img_delaunay = img.copy()

# Draw Delaunay triangles
for triangle in dt:
    pt1 = hull[triangle[0]]
    pt2 = hull[triangle[1]]
    pt3 = hull[triangle[2]]
    cv2.line(img_delaunay, pt1, pt2, (255, 0, 0), 1)
    cv2.line(img_delaunay, pt2, pt3, (255, 0, 0), 1)
    cv2.line(img_delaunay, pt3, pt1, (255, 0, 0), 1)

# Save the image with Delaunay triangulation
cv2.imwrite('delaunay_triangulation.png', img_delaunay)
