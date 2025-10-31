# 🧠 IntelliWatch-AI

**IntelliWatch-AI** is a real-time intelligent surveillance system that combines **human segmentation**, **multi-object tracking**, and **behavior analysis** into one unified pipeline.  
It uses **pretrained state-of-the-art models** and demonstrates efficient multi-model integration for **smart video understanding** — all wrapped in a **FastAPI** application and deployed on **Hugging Face Spaces**.

---

## 🚀 Features

- **Human Segmentation:** Powered by [SegFormer-B1](https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512) for pixel-accurate human masks.  
- **Multi-Object Tracking:** Integrated [ByteTrack](https://github.com/ifzhang/ByteTrack) for stable and real-time human tracking.  
- **Behavior Recognition:** Utilizes [VideoMAE](https://huggingface.co/MCG-NJU/videomae-base) pretrained on Kinetics for human action classification.  
- **Unified Pipeline:** Processes segmentation, tracking, and behavior recognition sequentially in real-time.  
- **API-Ready:** Fully deployed using **FastAPI** with user-friendly documentation and interactive endpoints.  
- **No Training Required:** Entirely based on pretrained models for high performance and simplicity.  

---

## 🧩 System Architecture

```text
Video Input
   │
   ├──► SegFormer-B1 → Human Segmentation
   │
   ├──► ByteTrack → Multi-Object Tracking
   │
   ├──► VideoMAE → Behavior Recognition
   │
   └──► FastAPI Endpoint → Processed Video Output

