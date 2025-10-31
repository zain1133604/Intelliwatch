# ============================================
# 🧠 Smart CCTV: Segmentation + Tracking + Behavior Recognition
# ============================================
# 📦 Install dependencies if not installed:
# pip install transformers torch torchvision opencv-python tqdm ultralytics

import torch
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import VideoMAEForVideoClassification,VideoMAEFeatureExtractor

# -------------------------------
# 1️⃣  Setup device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Using device: {device}")

# -------------------------------
# 2️⃣  Load Models
# -------------------------------
# Semantic Segmentation (SegFormer)
seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b1-finetuned-ade-512-512"
).to(device).eval()

# Object Detection + Tracking (YOLOv8 + ByteTrack)
detector = YOLO("yolov8n.pt")  # lightweight & fast

# Behavior Recognition (VideoMAE)
act_processor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
act_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(device).eval()

# -------------------------------
# 3️⃣  Load Video
# -------------------------------
video_path = r"A:\pel\kiti.mp4"  # change this path
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎞️ Total frames: {total_frames} | FPS: {fps}")

# -------------------------------
# 4️⃣  Output Video Setup
# -------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("smart_cctv_output.mp4", fourcc, fps, (width, height))

# -------------------------------
# 5️⃣  Buffers for Per-ID Action Clips
# -------------------------------
action_clip_len = 16
id_clip_buffers = {}

# -------------------------------
# 6️⃣  Process Each Frame
# -------------------------------
for _ in tqdm(range(total_frames), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Segmentation ---
    inputs = seg_processor(images=frame_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        seg_logits = seg_model(**inputs).logits

    pred_mask = torch.argmax(seg_logits, dim=1).squeeze().cpu().numpy()
    human_mask = (pred_mask == 12).astype(np.uint8)  # ADE20K class 12 = person

    # --- Detection + Tracking (YOLO + ByteTrack) ---
    results = detector.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
    ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []

    # --- For each tracked person ---
    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
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
        if len(id_clip_buffers[track_id]) == action_clip_len:
            clip = id_clip_buffers[track_id]

            # Resize all frames to 224x224 (VideoMAE input requirement)
            clip_resized = [cv2.resize(f, (224, 224)) for f in clip]

            # Convert to tensor input for VideoMAE
            inputs = act_processor(clip_resized, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = act_model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            label = act_model.config.id2label[pred]
        else:
            label = "Analyzing..."

        # --- Draw box + label ---
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {int(track_id)}: {label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --- Segmentation Overlay (FIXED version) ---
    mask_colored = cv2.applyColorMap((human_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask_colored = cv2.resize(mask_colored, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Blend safely (same shape)
    overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

    out.write(overlay)

# -------------------------------
# 7️⃣  Cleanup
# -------------------------------
cap.release()
out.release()
print("✅ Done! Video saved as 'smart_cctv_output.mp4'")



