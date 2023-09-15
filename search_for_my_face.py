import cv2
from flask import Flask, render_template, Response, stream_with_context
import mediapipe as mp
import numpy as np
import time 
import pandas as pd
import faiss
import os
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

POSE_TASK = None
TASK_COMPLETED = False
TASK_COMPLETION_TIME = None
latest_cropped_face = None
face_processed = False
distance_str = -2
index_folder_path = "/Users/root1/Desktop/face_verif/verification_demo/indexes"
index_file_name = index_folder_path + "/sample_valid.index"

app = Flask(__name__)
def serve_image(filename):
    image = cv2.imread(filename)
    _, jpeg = cv2.imencode('.jpg', image)
    response = Response(response=jpeg.tobytes(), content_type='image/jpeg')
    return response.data

def generate_ROI():
    global latest_cropped_face, face_processed
    cap = cv2.VideoCapture(1) 
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height
    
    mp_face_mesh = mp.solutions.face_mesh
    
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces = 1
    )

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_RGB2BGR)
        frame.flags.writeable = False

        result = face_mesh.process(frame)

        if not result.multi_face_landmarks:
            continue

        landmarks = result.multi_face_landmarks[0]
        face_oval = mp_face_mesh.FACEMESH_FACE_OVAL

        df = pd.DataFrame(list(face_oval), columns=['p1', 'p2'])

        routes_idx = []
        p1 = df.iloc[0]['p1']
        p2 = df.iloc[0]['p2']

        for i in range(0, df.shape[0]):
            obj = df[df['p1'] == p2]
            p1 = obj['p1'].values[0]
            p2 = obj['p2'].values[0]

            route_idx = []
            route_idx.append(p1)
            route_idx.append(p2)
            routes_idx.append(route_idx)

        routes = []

        for s, t in routes_idx:
            source = landmarks.landmark[s]
            target = landmarks.landmark[t]

            relative_source = (int(frame.shape[1]) * source.x, int(frame.shape[0] * source.y))
            relative_target = (int(frame.shape[1]) * target.x, int(frame.shape[0] * target.y))

            routes.append(relative_source)
            routes.append(relative_target)

        if routes:
            mask = np.zeros((frame.shape[0], frame.shape[1]))
            mask = cv2.fillConvexPoly(mask, np.array(routes, dtype=np.int32), 1)
            mask = mask.astype(bool)

            out = np.zeros_like(frame)
            out[mask] = frame[mask]

            x, y, w, h = cv2.boundingRect(np.array(routes, dtype=np.int32))

            cropped_face = out[y:y+h, x:x+w]
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

            latest_cropped_face = cropped_face
            face_processed = True

            annotated_image = out[:, :, ::-1]
            ret, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not ret:
                continue
            
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = jpeg.tobytes()
            
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

def generate_similar_image():
    global face_processed, latest_cropped_face, distance_str, index_file_name
    database = []
    default_path = "/Users/root1/Desktop/face_verif/verification_demo/dataset/Validation/1888-2037"
    dimension = 1280
    img_pathes = extract_path(default_path)

    if os.path.exists(index_file_name):
        print("Loading indexes from file...")
        database = faiss.read_index(index_file_name)
        print("Indexes loaded!")
    else:
        print("Creating indexes...")
        database = create_indexes(default_path, dimension)
        faiss.write_index(database, index_file_name)
        print("Indexes created!")

    while not face_processed:
        time.sleep(0.5)  # wait for 0.5 seconds before checking again

    while database is not []:
        distance, ind = face_search(database, latest_cropped_face, 1)
        similar_image_path = img_pathes[ind[0][0]]

        distance_str = "Distance: " + str(distance[0][0])
        
        similar_image = serve_image(similar_image_path)
        
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + similar_image + b'\r\n\r\n')
        
        time.sleep(1)

def embed_image_from_ndarray(img):
    base_options = mp.tasks.BaseOptions(
        model_asset_path="/Users/root1/Desktop/face_verif/verification_demo/models/mobilenet_v3_large.tflite"
    )
    l2_normalize = True
    quantize = True
    options = mp.tasks.vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize
    )

    with mp.tasks.vision.ImageEmbedder.create_from_options(options) as embedder:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        embedding = embedder.embed(mp_image)
        embedding = embedding.embeddings[0].embedding

        return np.reshape(embedding, (1, len(embedding)))
    
def embed_image(img):
    base_options = mp.tasks.BaseOptions(
        model_asset_path="/Users/root1/Desktop/face_verif/verification_demo/models/mobilenet_v3_large.tflite"
    )
    l2_normalize = True
    options = mp.tasks.vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize
    )

    with mp.tasks.vision.ImageEmbedder.create_from_options(options) as embedder:
        embedding = embedder.embed(img)
        return embedding
    
def face_dataset_to_embeddings(dataset_path):
    embeddings = []

    for img_path in dataset_path:
        obj = mp.Image.create_from_file(img_path)
        result = embed_image(obj)
        embedding_array = result.embeddings[0].embedding

        embeddings.append(embedding_array)

    return np.array(embeddings)

def extract_path(default_path):
    dataset_path = []
    directories = os.listdir(default_path)

    for directory in directories:
        if(directory == ".DS_Store"):
            continue

        temp = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(directory, "SR305"), "Light_02_Mid"), "real_01"), "color"), "crop")

        inner_dir = os.path.join(default_path, temp)
        for img in os.listdir(inner_dir):
            dataset_path.append(os.path.join(inner_dir, img))
    
    return dataset_path


def create_indexes(default_path, dimension):
    img_pathes = extract_path(default_path)
    source_embeddings = face_dataset_to_embeddings(img_pathes)
    index = faiss.IndexFlatIP(dimension)

    index.add(source_embeddings)
    
    return index

def face_search(index, face_img, k):
    face_embedding = embed_image_from_ndarray(face_img)
    D, I = index.search(face_embedding, k)
    return D, I

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_ROI()), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/similar_image_feed')
def similar_image_feed():
    return Response((generate_similar_image()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_distance')
def get_distance():
    global distance_str
    return distance_str

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run(port=8000)