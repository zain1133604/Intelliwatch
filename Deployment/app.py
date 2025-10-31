import torch
import cv2
import numpy as np
import os
import tempfile
import shutil
from ultralytics import YOLO
# --- UPDATED IMPORTS for SegFormer PyTorch model ---
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation 
from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask 

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Using device: {DEVICE}")

# Set a writable directory for model caches on Hugging Face Spaces
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
os.makedirs('/tmp/hf_cache', exist_ok=True)

# --- 1️⃣ Load Models ---

# Semantic Segmentation (SegFormer PyTorch) <-- CHANGE HERE
SEG_MODEL_NAME = "nvidia/segformer-b1-finetuned-ade-512-512"
seg_processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_NAME)
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    SEG_MODEL_NAME
).to(DEVICE).eval()

# Object Detection + Tracking (YOLOv8 + ByteTrack)
detector = YOLO("yolov8n.pt")

# Behavior Recognition (VideoMAE)
act_processor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
act_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(DEVICE).eval()
ACTION_CLIP_LEN = 16

# This dictionary stores the last predicted label for each ID
id_last_labels = {} 

# -------------------------------
# 2️⃣ FastAPI Setup
# -------------------------------
app = FastAPI(title="Smart CCTV Video Processor")

# -------------------------------
# 3️⃣ Utility Functions
# -------------------------------

def cleanup_task(dir_path):
    """Utility to create a BackgroundTask for deleting the temp directory."""
    # Ignore errors (like PermissionError on Windows) during cleanup
    return BackgroundTask(shutil.rmtree, dir_path, ignore_errors=True)

# --- REMOVED: preprocess_segformer() and run_segformer_onnx() ---
# They are replaced by the functions below using the HuggingFace SegFormer pipeline.

def run_segformer_pytorch(frame_rgb: np.ndarray, original_shape) -> np.ndarray:
    """
    Runs PyTorch SegFormer inference and returns the human mask.
    This replaces the ONNX code.
    """
    # 1. Preprocess
    # The processor handles resizing, normalization, and tensor conversion automatically
    inputs = seg_processor(images=frame_rgb, return_tensors="pt").to(DEVICE)

    # 2. Inference
    with torch.no_grad():
        seg_logits = seg_model(**inputs).logits

    # 3. Post-process
    # Resize logits to original image size
    seg_logits = torch.nn.functional.interpolate(
        seg_logits,
        size=(original_shape[0], original_shape[1]), # H, W
        mode='bilinear',
        align_corners=False
    )

    pred_mask = torch.argmax(seg_logits, dim=1).squeeze().cpu().numpy()
    
    # ADE20K class 12 = person
    human_mask = (pred_mask == 12).astype(np.uint8) 
    
    return human_mask

def run_action_recognition(clip_buffer: list) -> str:
    """Runs VideoMAE on a list of RGB frames (clip_buffer)."""
    # Resize all frames to 224x224 (VideoMAE input requirement)
    clip_resized = [cv2.resize(f, (224, 224), interpolation=cv2.INTER_LINEAR) for f in clip_buffer]
    
    # Convert to tensor input for VideoMAE
    inputs = act_processor(clip_resized, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = act_model(**inputs)
    pred = outputs.logits.argmax(-1).item()
    label = act_model.config.id2label.get(pred, "Unknown Action")
    return label

# -------------------------------
# 4️⃣ Main API Endpoint
# -------------------------------

@app.get("/")
async def root():
    return {"message": "Smart CCTV Backend: Upload a video to /process-video/"}

@app.post("/process-video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    global id_last_labels 
    
    if not file.content_type.startswith('video/'):
        await file.close() 
        return {"error": "Invalid file type. Only video files are supported."}, 400

    # MANUAL TEMP DIR SETUP: Fixes PermissionError by controlling cleanup
    tmpdir = tempfile.mkdtemp()
    input_path = os.path.join(tmpdir, file.filename)
    output_path = os.path.join(tmpdir, "smart_cctv_output.mp4")
    
    cap, out = None, None
    try:
        # Save the uploaded file to the temporary directory
        with open(input_path, "wb") as buffer:
            # IMPORTANT: Stream the file content to the buffer
            while content := await file.read(1024 * 1024): # Read in chunks (1MB)
                buffer.write(content)
        
        await file.close() 

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            return {"error": "Could not open video file."}, 500

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width <= 0 or height <= 0:
            return {"error": "Invalid video dimensions."}, 500

        # Output Video Setup
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Buffers for Per-ID Action Clips
        id_clip_buffers = {}
        id_last_labels.clear() 
        action_clip_len = ACTION_CLIP_LEN

        # Process Each Frame
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_shape = frame.shape
            
            # --- Segmentation (PyTorch) --- <-- CHANGE HERE
            human_mask = run_segformer_pytorch(frame_rgb, original_shape)
            
            # --- Detection + Tracking (YOLO + ByteTrack) ---
            results = detector.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
            ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
            
            # --- For each tracked person ---
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                
                # Crop with a slight buffer
                buffer_px = 5
                y1 = max(0, y1 - buffer_px)
                y2 = min(height, y2 + buffer_px)
                x1 = max(0, x1 - buffer_px)
                x2 = min(width, x2 + buffer_px)
                
                person_crop = frame_rgb[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                # Maintain a clip buffer per person ID
                id_clip_buffers.setdefault(track_id, [])
                id_clip_buffers[track_id].append(person_crop)

                # Keep only latest N frames
                if len(id_clip_buffers[track_id]) > action_clip_len:
                    id_clip_buffers[track_id] = id_clip_buffers[track_id][-action_clip_len:]

                # --- Action Recognition when clip ready ---
                label = id_last_labels.get(track_id, "Analyzing...")
                
                if len(id_clip_buffers[track_id]) == action_clip_len:
                    # Run inference every N frames to save time/resources
                    if frame_count % (action_clip_len // 2) == 0: 
                        label = run_action_recognition(id_clip_buffers[track_id])
                        id_last_labels[track_id] = label
                    
                # --- Draw box + label ---
                color = (0, 255, 0) 
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id}: {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # --- Segmentation Overlay ---
            # NOTE: We no longer need to resize the mask here as the PyTorch function handles interpolation
            mask_colored = cv2.applyColorMap((human_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Blend safely 
            overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

            out.write(overlay)
            frame_count += 1

        # --- Cleanup (Before Return) ---
        cap.release()
        out.release()
        
        # --- Return the processed video file ---
        return FileResponse(
            path=output_path, 
            media_type='video/mp4', 
            filename=f"processed_{os.path.basename(input_path)}",
            # Use background task to delete the temporary folder AFTER the response is sent
            background=cleanup_task(tmpdir) 
        )

    except Exception as e:
        # Ensure cleanup on failure
        if cap: cap.release()
        if out: out.release()
        # Clean up the manually created temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)
        # Re-raise the exception for FastAPI/Uvicorn to handle
        raise e
