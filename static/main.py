import cv2
import mediapipe as mp
import numpy as np
import csv
import glob
# 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)


def get_facemesh_coords(face_landmarks, image):
    h, w = image.shape[:2]  # grab width and height from image
    xyz = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
    return np.multiply(xyz, [w, h, w]).astype(int)

# For static images:
images_path = "static/image/"
images = glob.glob(images_path + "*.jpg")
IMAGE_FILES = []
for item in images:
    IMAGE_FILES.append(item)
# print(IMAGE_FILES)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    # for coordinate
    coordinate_path = "static/results/coordinate/"
    for face_landmarks in results.multi_face_landmarks:
        coords =   get_facemesh_coords(face_landmarks, image)
        save_path = coordinate_path + "image" + str(idx) + ".csv"
        with open(save_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(coords)
    # for mesh 
    mesh_path = "static/results/mesh/"  
    for face_landmarks in results.multi_face_landmarks:        
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
    save_path = mesh_path +'image' + str(idx) + '.png'
    cv2.imwrite(save_path, annotated_image)