import cv2
import numpy as np
from anomaly_detector import detect_anomalies, detect_objects, draw_boxes

# Define polygonal ROI for pedestrian path (edit these points as per video)
roi_polygon = np.array([[100, 300], [500, 300], [500, 600], [100, 600]])

cap = cv2.VideoCapture("videos/accident_path.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)
    anomalies = detect_anomalies(detections, roi_polygon)
    output_frame = draw_boxes(frame, detections, anomalies)

    # Draw ROI
    cv2.polylines(output_frame, [roi_polygon], isClosed=True, color=(255, 255, 0), thickness=2)

    cv2.imshow("Anomaly Detection", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
