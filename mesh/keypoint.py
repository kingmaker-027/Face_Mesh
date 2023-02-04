import csv
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# keypoints = [33,263,1,61,291,199]
keypoints = []
for i in range(0,468):
    keypoints.append(i)

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
            contour_point_idxs = []
            for point in mp_face_mesh.FACEMESH_CONTOURS:
                contour_point_idxs.append(point[0])
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in keypoints:
                        height, width = image.shape[:2]
                        point = (int(landmark.x * width), int(landmark.y * height))
                        cv2.circle(image, point, 1, (0, 255, 0), -1)
                        cv2.putText(image, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
cap.release()
