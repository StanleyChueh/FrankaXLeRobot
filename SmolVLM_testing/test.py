import cv2
import torch
import textwrap
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np

# Configuration
VIDEO_PATH = "ADIP_Smolvlm_test.mp4"
OUTPUT_VIDEO_PATH = "output_video.mp4"

MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device detected: {DEVICE}")

# Load model
print(f"Loading {MODEL_ID}...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map=DEVICE
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

if hasattr(processor, "image_processor"):
    processor.image_processor.do_resize = True
    processor.image_processor.size = {"height": 384, "width": 384}

print("Model Loaded! Starting Video Processing...")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Main loop: process every nth frame
frame_interval = 10  # Process every 10th frame
current_frame = 0
last_text = "" 

with torch.inference_mode():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video reached.")
            break

        # Only do AI inference every N frames
        if current_frame % frame_interval == 0:
            resized = cv2.resize(frame, (384, 384))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": "Describe the action in this video briefly."}
                    ]
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(DEVICE, dtype=torch.bfloat16)

            generated_ids = model.generate(**inputs, max_new_tokens=30)
            last_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(f"\rAI Says: {last_text}", end=" " * 20)

            del inputs, generated_ids
            torch.cuda.empty_cache()

        # Draw overlay EVERY FRAME using last_text
        display_frame = frame.copy()

        cv2.rectangle(display_frame, (0, 0), (frame_width, 150), (0, 0, 0), -1)
        wrapped = textwrap.wrap(last_text, width=50)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(wrapped):
            y = 40 + i * 35
            cv2.putText(display_frame, line, (20, y), font, 0.8, (0, 255, 0), 2)

        out.write(display_frame)

        # Show window frame
        cv2.imshow("Video Analysis", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_frame += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed. Output saved to:", OUTPUT_VIDEO_PATH)
