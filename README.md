# Ultra Light Live Camera Vision Language System using BLIP

# Abstract

This project presents a lightweight real-time Vision-Language Model (VLM) system for live scene understanding using a webcam stream. The system leverages the BLIP (Bootstrapping Language-Image Pretraining) image captioning model to generate natural language descriptions from live camera frames.

Unlike large-scale multimodal systems requiring high-end GPUs, this implementation is optimized for CPU-based inference, making it suitable for low-resource environments, edge devices, and academic experimentation.

The architecture emphasizes efficient preprocessing, background inference threading, and structured UI rendering to ensure smooth real-time interaction.

# 1. Introduction

Vision-Language Models (VLMs) bridge visual perception and natural language generation. They enable machines to describe scenes, understand context, and interact multimodally.

This project demonstrates:

*Real-time image-to-text generation

*CPU-based multimodal inference

*Threaded architecture for non-blocking execution

*Lightweight deployment using open-source frameworks

The system is designed as a compact and educational prototype for applied multimodal AI.

# 2. Model Architecture

This system uses the BLIP Image Captioning Base model developed by Salesforce.

**Model ID**: Salesforce/blip-image-captioning-base

**Framework**: PyTorch

**Precision**: Float32

**Inference Mode**: Evaluation (no gradient computation)

BLIP consists of:

*Vision Transformer (ViT) encoder for visual feature extraction

*Text decoder for autoregressive caption generation

*Cross-modal alignment mechanism

Caption generation is performed using beam search with:

*max_length = 50

*num_beams = 3

This configuration balances caption quality and inference latency.

# 3. System Architecture

Pipeline Overview

<img width="573" height="392" alt="image" src="https://github.com/user-attachments/assets/23057c10-6f18-4638-a98f-f6b53f4c1e64" />

**Architectural Design Choices**

*Background threading for inference

*CPU-only deployment

*Frame resizing to reduce memory footprint

*Gradient-disabled inference (torch.no_grad())

*Controlled thread usage via torch.set_num_threads(4)

This ensures smooth UI responsiveness while inference runs asynchronously.

# 4. Implementation Details

**Core Libraries**

*torch – Deep learning inference

*transformers – Model loading and preprocessing

*opencv-python – Video capture and UI rendering

*Pillow – Image conversion

*threading – Concurrent execution

**Performance Optimization Techniques**

*Thumbnail resizing to 384×384

*Beam search limited to 3 beams

*Float32 precision for compatibility

*Evaluation mode (model.eval())

*Inference every 4 seconds (configurable interval)

# 5. Hardware Requirements

**Minimum**

*4-core CPU

*8 GB RAM

*1 GB storage (model + cache)

*Webcam

**Recommended**

*8-core CPU

*16 GB RAM

*Optional GPU (change .to("cpu") → .to("cuda"))

**Model Size**

~500MB (downloaded automatically on first run)

# 6. Installation
   
**Clone Repository**

git clone https://github.com/your-username/ultra-light-live-camera-vlm.git

cd ultra-light-live-camera-vlm

**Create Virtual Environment**

python -m venv venv

venv\Scripts\activate

**Install Dependencies**

pip install -r requirements.txt

# 7. Requirements

torch

transformers

pillow

opencv-python

# 8. Execution

python blip2.py

**Controls**

* q → Quit

* s → Save frame

* SPACE → Manual analysis trigger

Automatic captioning occurs every 4 seconds.

# 9. Experimental Observations

On a standard 4-core CPU:

* Average inference time: 2–5 seconds per frame

* Memory usage: ~1.2–1.8 GB during execution

* Stable UI responsiveness due to threaded execution

Latency varies based on processor speed and background system load.

# 10. Applications

* Assistive AI systems

* Smart classroom demos

* Low-resource surveillance prototyping

* Human–AI interaction research

* Edge-based multimodal experimentation

# 11. Limitations

* CPU inference latency

* No object-level grounding

* No temporal reasoning across frames

* English-only captioning

# 12. Future Work

*GPU acceleration support

*Integration with LLaVA or Moondream for richer reasoning

*Object detection + caption fusion

*Streaming API endpoint

*Multilingual caption generation

*Performance benchmarking across hardware profiles

# 13. Author

Selin Jogi Chittilappilly

B.Voc Mathematics & Artificial Intelligence

AI Intern
