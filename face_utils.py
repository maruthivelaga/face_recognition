import os
import cv2
import numpy as np
from deepface import DeepFace
from numpy.linalg import norm

# Directories
KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_DIR = "embeddings"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def extract_embedding(image):
    try:
        emb = DeepFace.represent(img_path=image, model_name='Facenet', enforce_detection=False)[0]["embedding"]
        return np.array(emb)
    except:
        return None

def save_embeddings(name, embedding):
    np.save(os.path.join(EMBEDDINGS_DIR, f"{name}.npy"), embedding)

def load_known_embeddings():
    embeddings = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            embeddings[name] = np.load(os.path.join(EMBEDDINGS_DIR, file))
    return embeddings
def match_face(embedding, known_faces, threshold):
    best_match = ("Unknown", 1.0)
    
    for name, known_embedding in known_faces.items():
        # Cosine similarity
        similarity = np.dot(embedding, known_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(known_embedding))
        distance = 1 - similarity
        
        if distance < threshold and distance < best_match[1]:
            best_match = (name, distance)
    
    return best_match

def register_new_face(face_img, name):
    aligned = face_img
    embedding = extract_embedding(aligned)

    if embedding is not None:
        save_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        cv2.imwrite(save_path, aligned)
        save_embeddings(name, embedding)
        print(f"[✔] Registered new face: {name}")
    else:
        print("[✘] Failed to extract embedding.")