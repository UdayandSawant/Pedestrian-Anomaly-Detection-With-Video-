import cv2
import torch
import numpy as np
from collections import deque

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.conf = 0.4

# Define vehicle classes (YOLOv5 class names)
vehicle_classes = ['car', 'truck', 'bus', 'motorbike']

# Store previous positions of vehicles to detect movement
vehicle_tracks = {}
track_history = 10  # frames

# Thresholds
proximity_threshold = 50  # pixels
stationary_threshold = 3  # minimal movement across 3 frames

def detect_objects(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    return results.pandas().xyxy[0]

def get_center(det):
    return int((det['xmin'] + det['xmax']) / 2), int((det['ymin'] + det['ymax']) / 2)

def update_tracks(detections):
    global vehicle_tracks
    new_tracks = {}

    for _, det in detections.iterrows():
        if det['name'] in vehicle_classes:
            cx, cy = get_center(det)

            matched_id = None
            for vid, track in vehicle_tracks.items():
                prev_cx, prev_cy = track[-1]
                if abs(cx - prev_cx) < proximity_threshold and abs(cy - prev_cy) < proximity_threshold:
                    matched_id = vid
                    break

            if matched_id:
                new_tracks[matched_id] = vehicle_tracks[matched_id]
                new_tracks[matched_id].append((cx, cy))
                if len(new_tracks[matched_id]) > track_history:
                    new_tracks[matched_id].popleft()
            else:
                new_id = len(vehicle_tracks) + 1
                new_tracks[new_id] = deque([(cx, cy)], maxlen=track_history)

    vehicle_tracks = new_tracks

def detect_accidents():
    accidents = []
    for vid1, track1 in vehicle_tracks.items():
        if len(track1) < track_history:
            continue

        # Check if vehicle is mostly stationary
        x_movements = [abs(track1[i][0] - track1[i-1][0]) for i in range(1, len(track1))]
        y_movements = [abs(track1[i][1] - track1[i-1][1]) for i in range(1, len(track1))]
        if sum(x_movements) < 10 and sum(y_movements) < 10:
            for vid2, track2 in vehicle_tracks.items():
                if vid1 == vid2 or len(track2) < track_history:
                    continue

                # Check proximity to another stationary vehicle
                dist = np.linalg.norm(np.array(track1[-1]) - np.array(track2[-1]))
                if dist < 2 * proximity_threshold:
                    accidents.append((track1[-1], track2[-1]))
    return accidents

def draw(frame, detections, accidents):
    for _, det in detections.iterrows():
        cx, cy = get_center(det)
        color = (0, 0, 255) if det['name'] in vehicle_classes else (0, 255, 0)
        cv2.rectangle(frame, (int(det['xmin']), int(det['ymin'])), (int(det['xmax']), int(det['ymax'])), color, 2)
        cv2.putText(frame, det['name'], (int(det['xmin']), int(det['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (cx, cy), 3, color, -1)

    for pt1, pt2 in accidents:
        cv2.line(frame, pt1, pt2, (255, 0, 0), 3)
        cv2.putText(frame, 'Accident Detected', pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame

cap = cv2.VideoCapture("videos/accident_path.mp4")  # or path to a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)
    update_tracks(detections)
    accidents = detect_accidents()
    frame = draw(frame, detections, accidents)

    cv2.imshow("Road Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
