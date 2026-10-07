import cv2
import math
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = "sample.mp4"       # Path to input video (will be downloaded)
OUTPUT_PATH = "output.mp4"      # Path to save annotated video
MODEL_NAME = "yolov8n.pt"       # Lightweight YOLOv8 model

# Heuristic Thresholds
HISTORY_FRAMES = 15             # How many frames to look back for speed calculation
SPEED_THRESHOLD = 30.0          # Pixels moved in HISTORY_FRAMES to be considered "Running"
INACTIVITY_DISTANCE = 10.0      # Must stay within this radius to be considered "Inactive"
INACTIVITY_FRAMES_MAX = 90      # Number of consecutive frames (e.g. 3s @ 30fps) to trigger injury alert

# State Tracking
# track_history[track_id] = [(x, y), (x, y), ...]
track_history = defaultdict(list)
# inactivity_counter[track_id] = integer
inactivity_counter = defaultdict(int)

def main():
    print(f"Loading YOLO model: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}. Make sure it exists.")
        return

    # Get video properties for saving
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30 # Default if OpenCV fails to read it

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    frame_count = 0

    print("Starting video processing...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1

        # Run YOLO tracking (persist=True is required for ByteTrack assigning IDs)
        # We only want class 0 (person)
        results = model.track(frame, persist=True, classes=[0], verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                # Calculate center of the bounding box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Update history
                track_history[track_id].append((cx, cy))
                
                # Default state
                behavior = "Normal"
                color = (0, 255, 0) # Green
                
                if len(track_history[track_id]) > HISTORY_FRAMES:
                    # Remove old history
                    track_history[track_id].pop(0)
                    
                    # Calculate speed (distance over HISTORY_FRAMES)
                    old_cx, old_cy = track_history[track_id][0]
                    dist = math.hypot(cx - old_cx, cy - old_cy)
                    
                    # 1. Check for Running
                    if dist > SPEED_THRESHOLD:
                        behavior = "RUNNING (Hazard)"
                        color = (0, 0, 255) # Red
                        print(f"[{frame_count/fps:.1f}s] WARNING: Person #{track_id} detected running.")
                        inactivity_counter[track_id] = 0 # Reset inactivity
                        
                    # 2. Check for Inactivity
                    elif dist < INACTIVITY_DISTANCE:
                        inactivity_counter[track_id] += 1
                        if inactivity_counter[track_id] > INACTIVITY_FRAMES_MAX:
                            behavior = "INACTIVE (Possible Injury)"
                            color = (0, 0, 255) # Red
                            # Only print once every second while inactive so it doesn't spam
                            if inactivity_counter[track_id] % fps == 0:
                                print(f"[{frame_count/fps:.1f}s] CRITICAL: Person #{track_id} inactive for {(inactivity_counter[track_id]/fps):.1f}s.")
                    else:
                        # Normal movement
                        inactivity_counter[track_id] = 0

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {track_id} | {behavior}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw tracking trail
                points = np.array(track_history[track_id]).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processing complete! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
