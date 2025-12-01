import cv2
import time
import torch
import textwrap 
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np

# Model and Device Configuration
MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Model
print(f"Loading {MODEL_ID} on {DEVICE}...")
try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", 
        device_map=DEVICE
    )
except ValueError:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa",
        device_map=DEVICE
    )

processor = AutoProcessor.from_pretrained(MODEL_ID)

# 2. Optimize Processor
if hasattr(processor, "image_processor"):
    processor.image_processor.do_resize = True
    processor.image_processor.size = {"height": 512, "width": 512}

print("Model Loaded! Starting Camera...")

def get_latest_frames(cap, num_frames=4):
    frames = []
    for _ in range(5): cap.grab() # Flush buffer
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (640, 480)) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb_frame))
        time.sleep(0.02) 
        
    return frames

# 3. Main Loop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

try:
    with torch.inference_mode(): 
        while True:
            frames = get_latest_frames(cap, num_frames=4)
            if not frames: continue

            # Get the image for display (convert back to BGR for OpenCV)
            debug_frame = cv2.cvtColor(np.array(frames[-1]), cv2.COLOR_RGB2BGR)
            
            # Inference
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f} for f in frames
                    ] + [{"type": "text", "text": "Describe the action in this video briefly."}]
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(DEVICE, dtype=torch.bfloat16)

            generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=24)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Print to console
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (0, 255, 0) # Green
            thickness = 2
            position = (10, 30)

            # Wrap text
            wrapped_text = textwrap.wrap(generated_text, width=40)

            for i, line in enumerate(wrapped_text):
                y_offset = position[1] + (i * 25)
                cv2.putText(debug_frame, line, (position[0], y_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)

            # Show result
            cv2.imshow("What the AI Sees", debug_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # Cleanup
            del inputs, generated_ids, messages
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    cap.release()
    cv2.destroyAllWindows()
