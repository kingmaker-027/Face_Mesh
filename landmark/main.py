import cv2
import mediapipe as mp
import numpy as np
import csv

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def get_facemesh_coords(face_landmarks, image):
    h, w = image.shape[:2]  
    xyz = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
    return np.multiply(xyz, [w, h, w]).astype(int)

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
     max_num_faces=1,
     refine_landmarks=True,
     min_detection_confidence=0.5,
     min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image.flags.writeable = False
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            coordinate_path = "landmark/"
            for face_landmarks in results.multi_face_landmarks:
                coords =   get_facemesh_coords(face_landmarks, image)
                print(coords)
                save_path = coordinate_path + "landmarks"".csv"
                with open(save_path, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(coords)
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

