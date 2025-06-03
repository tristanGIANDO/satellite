from PIL import Image
import numpy as np
import os
import rasterio

def save_mask_as_png_pil(mask: np.ndarray, path: str, threshold: float = 0.5):
    """
    Sauvegarde un masque binaire en PNG à partir d'un array NumPy, via Pillow.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    bin_mask = (mask > threshold).astype(np.uint8) * 255  # 0 ou 255
    img = Image.fromarray(bin_mask, mode="L")  # 'L' = grayscale 8-bit
    img.save(path)


def save_rgba_as_png(rgba: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert to uint8 for saving
    rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    img.save(path)


def create_rgba(dataset, mask: np.ndarray, image_idx: int = 2) -> np.ndarray:
    """
    Crée une image RGBA en utilisant les chemins RGB depuis dataset.data,
    et le masque de prédiction fourni.
    """
    row = dataset.data.iloc[image_idx]
    rgb_paths = {band: row[band] for band in ["red", "green", "blue"]}

    def normalize_stretch(img, min_percent=2, max_percent=98):
        min_val = np.percentile(img, min_percent)
        max_val = np.percentile(img, max_percent)
        return np.clip((img - min_val) / (max_val - min_val), 0, 1)

    channels = []
    for band in ["red", "green", "blue"]:
        with rasterio.open(rgb_paths[band]) as src:
            img = src.read(1).astype(np.float32) / 65535.0
            img = normalize_stretch(img)  # Normalisation pour étirer les valeurs
            channels.append(img)

    rgb = np.stack(channels, axis=-1)  # (H, W, 3)
    H_mask, W_mask = mask.shape
    rgb = rgb[:H_mask, :W_mask, :]  # crop pour correspondre au masque

    alpha_inverted = 1.0 - mask  # inverse du masque : +transparent si prédiction forte
    alpha_inverted = np.clip(alpha_inverted, 0.0, 1.0)

    rgba = np.concatenate([rgb, alpha_inverted[..., None]], axis=-1)  # (H, W, 4)
    
    return rgba