import cv2
import mediapipe as mp
import numpy as np
import csv
import glob

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def get_facemesh_coords(face_landmarks, image):
    h, w = image.shape[:2]  # grab width and height from image
    xyz = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
    return np.multiply(xyz, [w, h, w]).astype(int)

cap = cv2.VideoCapture(0)
i = 0
# a variable to set how many frames you want to skip
frame_skip = 30
# a variable to keep track of the frame to be saved
frame_count = 0
store_path = "live_cap/image/"
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Live Capture", frame)
    if i > frame_skip - 1:
        frame_count += 1
        save_path = store_path+'img'+str(frame_count) + '.jpg'
        cv2.imwrite(save_path, frame)
        i = 0
        continue
    i += 1
    if frame_count > 10:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
images_path = "live_cap/image/"
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
    coordinate_path = "live_cap/results/coordinate/"
    for face_landmarks in results.multi_face_landmarks:
        coords =   get_facemesh_coords(face_landmarks, image)
        save_path = coordinate_path + "image" + str(idx) + ".csv"
        with open(save_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(coords)
    # for mesh 
    mesh_path = "live_cap/results/mesh/"  
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


