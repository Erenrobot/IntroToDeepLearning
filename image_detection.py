from ultralytics import YOLO
import cv2
# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Define path to the image file
source = 'deneme.png'

# Run inference on the source
results = model(source)  # list of Results objects
annotated_frame = results[0].plot()

# Display the annotated frame
cv2.imshow("YOLOv8 Inference", annotated_frame)
cv2.waitKey(0)