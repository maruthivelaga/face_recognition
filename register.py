# register.py
import cv2
import os
from face_utils import extract_embedding, save_embeddings, KNOWN_FACES_DIR

def register_new_face():
    # Take user input before initializing camera
    name = input("Enter name for registration: ").strip()
    if not name:
        print("[-] Invalid name. Registration aborted.")
        return None

    # Create directory if it doesn't exist
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    # Check for existing registration
    existing_files = [f for f in os.listdir(KNOWN_FACES_DIR) if f.startswith(name)]
    if existing_files:
        print(f"[-] Name '{name}' already exists in the database.")
        overwrite = input("Do you want to overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("[INFO] Registration cancelled.")
            return None

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Could not open camera.")
        return None

    # Create named window first and move it to foreground
    cv2.namedWindow("Register Face", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Register Face", 800, 600)
    
    # Try to bring window to front (OS dependent)
    try:
        if os.name == 'nt':  # Windows
            import ctypes
            ctypes.windll.user32.ShowWindow(ctypes.windll.user32.GetActiveWindow(), 3)
    except:
        pass

    print("[INFO] Press 's' to capture face, 'q' to quit.")
    print("[INFO] Make sure your face is well-lit and clearly visible.")

    captured = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[-] Failed to capture frame.")
            continue

        # Mirror the frame for more natural viewing
        frame = cv2.flip(frame, 1)

        # Display instructions on the frame
        cv2.putText(frame, "Press 's' to capture", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Register Face", frame)
        
        # Bring window to front (another attempt)
        cv2.setWindowProperty("Register Face", cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty("Register Face", cv2.WND_PROP_TOPMOST, 0)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Attempt to extract face embedding
            embedding = extract_embedding(frame)
            if embedding is not None:
                save_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
                try:
                    cv2.imwrite(save_path, frame)
                    save_embeddings(name, embedding)
                    print(f"[✓] Face successfully registered for: {name}")
                    captured = True
                except Exception as e:
                    print(f"[-] Error saving files: {e}")
            else:
                print("[-] No face detected or failed to extract embedding. Try again.")
            break

        elif key == ord('q'):
            print("[INFO] Registration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return name if captured else None

if __name__ == "__main__":
    registered_name = register_new_face()
    if registered_name:
        print(f"Registration complete for {registered_name}")