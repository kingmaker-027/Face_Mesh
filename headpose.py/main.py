import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import csv

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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

        img_h, img_w, img_d = image.shape
        face_3d = []
        face_2d = []
        # keypoints = [33,263,1,61,291,199]
        keypoints = []
        for i in range(0,468):
            keypoints.append(i)
        if results.multi_face_landmarks:
            for face_landmark in results.multi_face_landmarks:
                for idx,lm in enumerate(face_landmark.landmark):
                    if idx in keypoints:
                        if idx == 1:
                            key_1_2d = (lm.x * img_w, lm.y * img_h)
                            key_1_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3) 
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z]) 
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * img_w
                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                #key_1
                nose_3d_projection, jacobian = cv2.projectPoints(key_1_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(key_1_2d[0]), int(key_1_2d[1]))
                p2 = (int(key_1_2d[0] + y * 5) , int(key_1_2d[1] - x * 5))
                cv2.line(image, p1, p2, (255, 0, 0), 1)


        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
cap.release()

          