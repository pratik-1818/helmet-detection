from ultralytics import YOLO

# Load the YOLOv8 model (you can start with a pre-trained model like YOLOv8n)
model = YOLO('yolov8m.pt')  # Or choose a different variant

# Train the model
model.train(
    data='/home/vmukti/Desktop/helmet/data.yaml',  # Path to your data.yaml file
    epochs=50,  # Number of epochs
    imgsz=640,  # Image size (e.g., 640x640)
    batch=16,  # Batch size
    name='helmet-detection'  # Experiment name
)
