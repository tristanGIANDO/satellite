import matplotlib.pyplot as plt
import numpy as np

def show_prediction(full_mask: np.ndarray, threshold=0.5):
    binary = (full_mask > threshold).astype(float)
    plt.figure(figsize=(10, 10))
    plt.imshow(binary, cmap="gray")
    plt.title("Masque reconstruit")
    plt.axis("off")
    plt.show()
