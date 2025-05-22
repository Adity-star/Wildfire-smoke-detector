import gradio as gr
from ultralytics import YOLO
from PIL import Image
import torch
import os

# Load YOLOv8 model
MODEL_PATH = "models/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(DEVICE)

# Detection function
def detect_smoke(image, conf_thresh):
    if image is None:
        return None, "Please upload or capture an image."

    image = image.convert("RGB")
    results = model.predict(source=image, conf=conf_thresh, device=DEVICE)

    # Annotated image with bounding boxes
    annotated_img = results[0].plot()
    pil_annotated = Image.fromarray(annotated_img)

    # Extract detected classes
    boxes = results[0].boxes
    if boxes is not None and boxes.cls.numel() > 0:
        labels = [model.names[int(cls)] for cls in boxes.cls]
        label_summary = f"🔥 {len(labels)} detection(s): " + ", ".join(set(labels))
    else:
        label_summary = "✅ No wildfire smoke detected."

    return pil_annotated, label_summary

# Gradio interface
title = "🔥 Wildfire Smoke Detector with YOLOv8"
description = """
Upload an image or use your webcam to detect signs of wildfire smoke using a YOLOv8 object detection model.
Adjust the confidence threshold for sensitivity. Results will be shown with bounding boxes and detection summary.
"""

examples = [[r"C:\Users\Administrator\OneDrive\Desktop\Wildfire-smoke-detector\data\valid\images\ck0k9lqaz4ict0863typf3ngd_jpeg.rf.6a740fd0c445713ba9ab596507348319.jpg", 0.25], [r"C:\Users\Administrator\OneDrive\Desktop\Wildfire-smoke-detector\data\valid\images\ck0kexd04j9w30a4628x0rzxi_jpeg.rf.6a174120a844ca0a1362c1ba4ce9dae5.jpg", 0.25]]

demo = gr.Interface(
    fn=detect_smoke,
    inputs=[
        gr.Image(type="pil", label="Input Image", sources=["upload", "webcam", "clipboard"]),
        gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(type="pil", label="Detection Result"),
        gr.Textbox(label="Detection Summary")
    ],
    title=title,
    description=description,
    examples=examples,
    theme="default"
)

if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True)
