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


