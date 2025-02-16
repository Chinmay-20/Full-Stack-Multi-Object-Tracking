from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import cv2
import torch
import numpy as np
from uuid import uuid4
import sys
sys.path.append("/Users/chinmay/Documents/multi_object_tracking")
from object_tracking import Yolo_implmentation 
from time import time
from pymongo import MongoClient
import os 
import boto3
from dotenv import load_dotenv
import uuid
import mimetypes
import io
import tempfile
from tqdm import tqdm

load_dotenv()

# mongodb cloud connection
MONGO_URI = "mongodb://localhost:27017"

# AWS S3 Configuration
S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME")
S3_REGION = os.getenv("AWS_S3_REGION")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=S3_REGION
)
res = s3_client.list_buckets()
print("Buckets available:", [bucket["Name"] for bucket in res["Buckets"]])


client = MongoClient(MONGO_URI)
db = client["object_tracking_db"]
collection = db["tracking_results"]

# fastAPI connections
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize YOLO tracking model
yolo_tracker = Yolo_implmentation()

# Storage for tracking results
tracking_results = {}
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def upload_to_s3(file_path):
    """Upload processed media (image/video) to S3 from file path and return its URL."""
    file_name = os.path.basename(file_path)
    content_type = "video/mp4"
    
    with open(file_path, "rb") as file_data:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=file_name,
            Body=file_data,
            ContentType=content_type
        )
    
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{file_name}"

def process_file(file_path, file_id, file_name):
    """Process the uploaded image/video and return tracking results using object tracking."""
    results = []

    if file_path.endswith((".mp4", ".avi")):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file '{file_path}'. The file might be corrupted.")
        

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = os.path.join(UPLOAD_DIR, f"{file_id}_processed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame, frame_results = yolo_tracker.process_single_image(frame_rgb)
                results.append(format_results(frame_results))
                
                processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                out.write(processed_frame_bgr)
                
                pbar.update(1)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    s3_url = upload_to_s3(output_path)

    collection.insert_one({
        "file_id": file_id,
        "file_name": file_name,
        "s3_url": s3_url,
        "results": results
    })
    
    return s3_url


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1]
    if file_extension not in [ "mp4", "avi"]:
        return {"error": "Unsupported file format"}
    
    # Save the file
    file_id = str(uuid4())
    file_name = file.filename
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_extension}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process file using object tracking
    result = process_file(file_path, file_id, file_name)
    
    return {"file_id": file_id, "s3_url": result, "message": "File uploaded and processed successfully"}

    


@app.get("/results/{file_id}")
def get_results(file_id: str):
    result = collection.find_one({"file_id": file_id}, {"_id":0}) # Exclude mongodb's _id field 

    if not result:
        raise HTTPException(status_code=404,  detail = "File ID not found")
    
    # Extract tracking statistics
    num_frames = len(result["results"])
    total_objects = sum(len(frame) for frame in result["results"])
    avg_objects_per_frame = total_objects / num_frames if num_frames else 0

    return {
        "file_id": file_id,
        "file_name": result["file_name"],
        "total_frames_processed": num_frames,
        "total_objects_detected": total_objects,
        "average_objects_per_frame": round(avg_objects_per_frame, 2)
    }

@app.get("/files")
def get_all_files():
    """Return a list of all stored file names and their corresponding file IDs."""
    files = collection.find({}, {"_id": 0, "file_id": 1, "file_name": 1})  # Get only file_id and file_name
    file_list = list(files)
    
    if not file_list:
        return {"message": "No files found in the database."}

    return {"files": file_list}


@app.get("/get-video/{file_id}")
def get_video(file_id: str):
    """Retrieve the S3 URL of a processed video using file_id."""
    file_entry = collection.find_one({"file_id": file_id})
    if not file_entry:
        raise HTTPException(status_code=404, detail="File not found")
    return {"s3_url": file_entry["s3_url"]}

@app.delete("/delete/{file_id}")
async def delete_file(file_id: str):
    """Delete a file entry from the database based on file_id."""
    # result = collection.delete_one({"file_id": file_id})

    # if result.deleted_count == 0:
    #     raise HTTPException(status_code=404, detail="File ID not found")

    # return {"message": "File successfully deleted"}
    file_entry = collection.find_one({"file_id": file_id})
    
    if not file_entry:
        raise HTTPException(status_code=404, detail="File ID not found")
    
    s3_url = file_entry.get("s3_url")
    if s3_url:
        s3_object_key = s3_url.split("/")[-1]  # Extract object key from S3 URL
        try:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_object_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete from S3: {str(e)}")
    
    result = collection.delete_one({"file_id": file_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="File ID not found")
    
    return {"message": "File successfully deleted from MongoDB and S3"}




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
