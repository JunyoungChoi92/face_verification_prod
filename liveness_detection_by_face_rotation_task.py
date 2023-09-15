import cv2
from flask import Flask, render_template, Response, stream_with_context
import mediapipe as mp
import numpy as np
import time 
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

POSE_TASK = None
TASK_COMPLETED = False
TASK_COMPLETION_TIME = None

app = Flask(__name__)

def generate_head_pose_estimation():
    global TASK_COMPLETED, TASK_COMPLETION_TIME, POSE_TASK

    cap = cv2.VideoCapture(1) 

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces = 1
    )

    update_pose_task()

    while cap.isOpened():
        ret, frame = cap.read()

        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        
        result = face_mesh.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_h, frame_w, frame_c = frame.shape
        mesh_idx = [33, 263, 1, 61, 291, 199]
        face_3d = []
        face_2d = []

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in mesh_idx:
                        if idx == 1:
                            nose_2d = (lm.x * frame_w, lm.y * frame_h)
                            nose_3d = (lm.x * frame_w, lm.y * frame_h, lm.z * 3000)
                        x, y = int(lm.x * frame_w), int(lm.y * frame_h)

                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * frame_w # Not zoomed in

                # called the camera intrinsic matrix or the camera matrix. 
                # It encodes the properties of a camera that relate the 3D world to the 2D image captured by the camera. 
                # This matrix is fundamental in computer vision and photogrammetry.
                cam_matrix = np.array([
                    [focal_length, 0, frame_w / 2],
                    [0, focal_length, frame_h / 2],
                    [0, 0, 1]
                ], dtype=np.float64)

                dist_matrix = np.zeros((4, 1), dtype=np.float64) # Assuming no lens distortion
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                if not success:
                    continue

                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y < -10:
                    detected_pose = "Looking: Left"
                elif y > 10:
                    detected_pose = "Looking: Right"
                elif x < -10:
                    detected_pose = "Looking: Down"
                elif x > 10:
                    detected_pose = "Looking: Up"
                else:
                    detected_pose = "Looking: Forward"
                
                if detected_pose == f"Looking: {POSE_TASK}":
                    cv2.putText(frame, "CORRECT", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    if not TASK_COMPLETED:
                        TASK_COMPLETED = True
                        TASK_COMPLETION_TIME = time.time()
                    
                    if time.time() - TASK_COMPLETION_TIME > 3:
                        update_pose_task()
                else:
                    cv2.putText(frame, "WRONG", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

                cv2.putText(frame, f"Task: Look {POSE_TASK}", (20,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                cv2.putText(frame, detected_pose, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                # p1 = (int(nose_2d[0]), int(nose_2d[1]))
                # p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                # cv2.line(frame, p1, p2, (0, 255, 0), 2)

                cv2.putText(frame, detected_pose, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    cap.release()

def update_pose_task():
    global POSE_TASK, TASK_COMPLETED
    poses = ["Right", "Left", "Up", "Down", "Forward"]
    POSE_TASK = np.random.choice(poses)
    TASK_COMPLETED = False

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_head_pose_estimation()), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/give_task')
def endow_ALD_task():
    update_pose_task()
    return f"Task: Turn your head {POSE_TASK}"

@app.route('/')
def index():
    return render_template('ALD.html')

if __name__ == '__main__':
    app.debug = True
    app.run(port=8000)