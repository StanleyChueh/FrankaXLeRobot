import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# psnr calculation
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

# load image
img = Image.open("512x512.png").convert("RGB")

to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()

img_t = to_tensor(img).unsqueeze(0)  # (1,3,H,W)

print("Original:", img_t.shape)

# pixel shuffle(downsample)
down_factor = 4   

down = F.pixel_unshuffle(img_t, down_factor)
print("After pixel-unshuffle:", down.shape)

# pixel shuffle(upsample)
up = F.pixel_shuffle(down, down_factor)
print("Reconstructed:", up.shape)

# Clip into valid range
up = torch.clamp(up, 0.0, 1.0)

# psnr calculation
value = psnr(img_t, up)
print("PSNR:", value, "dB")

# save recovered image
out_img = to_image(up.squeeze(0))
out_img.save("recovered.png")

print("Saved recovered image to recovered.png")
