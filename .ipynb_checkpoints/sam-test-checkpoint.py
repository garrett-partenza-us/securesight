import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

model = build_sam2(model_cfg, checkpoint, device='cpu')
# predictor = SAM2ImagePredictor(model)
mask_generator = SAM2AutomaticMaskGenerator(model)

image = Image.open('dog_bike_car.jpg')
image = np.array(image.convert("RGB"))

"""
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[216,363]]),
        point_labels=np.array([1]),
        multimask_output=True
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
"""

output = mask_generator.generate(image)

masks = np.vstack(list(mask['segmentation'] for mask in output))
scores = list(mask['predicted_iou'] for mask in output)
# Set colors for masks (Red, Green, Blue)
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # RGB

# Set up the plot with 3 subplots (one for each mask)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loop through each mask and create a subplot for each
for i in range(len(masks)):
    mask = masks[i]
    
    # Create a color mask using the chosen color (Red, Green, Blue)
    color_mask = np.zeros_like(image, dtype=np.float32)
    color_mask[mask == 1] = colors[i]  # Apply color to the mask area
    
    # Combine the original image with the colored mask (using alpha blending)
    masked_image = np.clip(image + (color_mask * 255), 0, 255).astype(np.uint8)
    
    # Plot the image with the mask on the respective subplot
    ax = axes[i]
    ax.imshow(masked_image)
    ax.axis('off')  # Hide axes for a cleaner display
    ax.set_title(f"Score: {scores[i]:.2f}")  # Title with score

# Adjust layout
plt.tight_layout()
plt.show()

# Optionally print the masks' shapes and scores
print("Masks shape:", masks.shape)
print("Scores shape:", scores.shape)
print("Logits shape:", logits.shape)
