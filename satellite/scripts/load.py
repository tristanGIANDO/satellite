import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

# Replace these with your actual file paths
r_file = "satellite_data/sentinel2/2025-04-02/31UDQ/B04.jp2"
g_file = "satellite_data/sentinel2/2025-04-02/31UDQ/B03.jp2"
b_file = "satellite_data/sentinel2/2025-04-02/31UDQ/B02.jp2"
n_file = "satellite_data/sentinel2/2025-04-02/31UDQ/B08.jp2"

# Open each band
with (
    rasterio.open(r_file) as src_r,
    rasterio.open(g_file) as src_g,
    rasterio.open(b_file) as src_b,
    rasterio.open(n_file) as src_n,
):
    r = src_r.read(1)
    g = src_g.read(1)
    b = src_b.read(1)
    n = src_n.read(1)  # Near-infrared band, not used in RGB but can be useful for other analyses

# Stack bands into an RGB image
rgbn = np.dstack((r, g, b, n))


def _gray_world_balance(stacked_image: np.ndarray) -> np.ndarray:
    # img: [H, W, C] in float32
    avg_per_channel = stacked_image.mean(axis=(0, 1))
    gray_avg = avg_per_channel.mean()
    scale = gray_avg / (avg_per_channel + 1e-6)

    return np.clip(stacked_image * scale, 0, 1)


# Stretch to 0-1 for display if needed
def stretch(img):
    img = img.astype(np.float32)
    img_min, img_max = np.percentile(img, (2, 98))
    img = np.clip((img - img_min) / (img_max - img_min), 0, 1)
    return img


rgbn = _gray_world_balance(stretch(rgbn))
# Display
# Save the RGB image if needed
import os

output_rgb_file = "satellite_data/sentinel2/2025-04-02/31UDQ/rgb_preview.png"
os.makedirs(os.path.dirname(output_rgb_file), exist_ok=True)

# Ensure the path is valid for your OS (especially on Windows)
output_rgb_file = os.path.normpath(output_rgb_file)

rgb_img = (rgbn[..., :3] * 255).astype(np.uint8)
Image.fromarray(rgb_img).save(output_rgb_file)
print(f"RGB preview saved to {output_rgb_file}")
