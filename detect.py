import cv2
import numpy as np
from deepface import DeepFace
from face_utils import load_known_embeddings, match_face

# Constants
MIN_FACE_SIZE = 100  # Minimum face size in pixels
MATCH_THRESHOLD = 0.55  # Lower is more strict
DISPLAY_SCALE = 0.7

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Camera error")
        return

    # Load known faces
    known_faces = load_known_embeddings()
    if not known_faces:
        print("❌ No registered faces. Run register.py first")
        return

    print(f"✅ Loaded {len(known_faces)} known faces")
    print("🔍 Starting detection... Press ESC to quit")

    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame error")
            break

        frame = cv2.flip(frame, 1)  # Mirror
        
        try:
            # Detect faces using OpenCV (more reliable than DeepFace for detection)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                if w < MIN_FACE_SIZE:
                    continue

                # Extract face
                face_img = frame[y:y+h, x:x+w]
                
                # Get embedding using DeepFace
                try:
                    embedding = DeepFace.represent(face_img, model_name='Facenet')[0]["embedding"]
                    
                    # Match against known faces
                    name, dist = match_face(embedding, known_faces, MATCH_THRESHOLD)
                    confidence = max(0, 1 - dist)
                    
                    # Draw results
                    color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
                    label = f"{name} ({confidence:.0%})" if confidence > 0.7 else "Unknown"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                except Exception as e:
                    print(f"⚠️ Recognition error: {e}")
                    continue

        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            continue

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()