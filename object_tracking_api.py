from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
import os
import cv2
import torch
import numpy as np
from uuid import uuid4
import sys
sys.path.append("/Users/chinmay/Documents/multi_object_tracking")

from object_tracking import Yolo_implmentation  # Import again


app = FastAPI()

# Initialize YOLO tracking model
yolo_tracker = Yolo_implmentation()

# Storage for tracking results
tracking_results = {}
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1]
    if file_extension not in ["jpg", "jpeg", "png", "mp4", "avi"]:
        return {"error": "Unsupported file format"}
    
    # Save the file
    file_id = str(uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_extension}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process file using object tracking
    result = process_file(file_path)
    tracking_results[file_id] = result
    
    return {"file_id": file_id, "message": "File uploaded and processed successfully"}

# @app.get("/results/{file_id}")
# def get_results(file_id: str):
#     if file_id in tracking_results:
#         return tracking_results[file_id]
#     return {"error": "File ID not found"}

# @app.get("/results/{file_id}")
# def get_results(file_id: str):
#     if file_id in tracking_results:
#         return json.dumps(tracking_results[file_id])  # Ensure JSON format
#     return {"error": "File ID not found"}

@app.get("/results/{file_id}")
def get_results(file_id: str):
    if file_id not in tracking_results:
        return {"error": "File ID not found"}

    results = tracking_results[file_id]
    
    # Extract tracking statistics
    num_frames = len(results)
    total_objects = sum(len(frame) for frame in results)
    avg_objects_per_frame = total_objects / num_frames if num_frames else 0

    return {
        "file_id": file_id,
        "total_frames_processed": num_frames,
        "total_objects_detected": total_objects,
        "average_objects_per_frame": round(avg_objects_per_frame, 2)
    }




def process_file(file_path):
    """Process the uploaded image/video and return tracking results using object tracking."""
    results = []
    
    if file_path.endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(file_path)
        results = yolo_tracker.process_single_image(img)

    elif file_path.endswith((".mp4", ".avi")):
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, frame_results = yolo_tracker.process_single_image(frame)
            results.append(format_results(frame_results))
        # print(results)
        cap.release()
    
    return results

def format_results(frame_results):
    """Convert numpy arrays and objects to JSON-friendly format."""
    formatted = []
    # for frame_data in results:
    #     img_array, obstacles = frame_data  # Unpack tuple
        
    formatted.append({
        # "frame": img_array.tolist(),  # Convert numpy array to list
        "obstacles": [format_obstacle(res) for res in frame_results]  # Convert Obstacle objects
    })
    return formatted

def format_obstacle(obstacle):
    """Convert Obstacle object to dictionary."""
    return {
        "id": obstacle.idx,
        "bbox": obstacle.box,
        "age": obstacle.age,
        "unmatched_age": obstacle.unmatched_age
    }
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
curl -X 'POST' \
  'http://localhost:8000/upload' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_video.mp4'


curl -X 'GET' \
  'http://localhost:8000/results/123e4567-e89b-12d3-a456-426614174000'


"""