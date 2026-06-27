from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


def make_contact_sheet(images: list[np.ndarray], path: Path, cols: int = 5) -> None:
    if not images:
        return
    rgb_images = []
    for image in images:
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_images.append(rgb)
    rows = int(np.ceil(len(rgb_images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.0))
    axes_arr = np.array(axes).reshape(-1)
    for ax, img in zip(axes_arr, rgb_images):
        ax.imshow(img)
        ax.axis("off")
    for ax in axes_arr[len(rgb_images) :]:
        ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_panel(path: Path, panels: dict[str, np.ndarray], cmap: str = "gray") -> None:
    if not panels:
        return
    cols = min(3, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes_arr = np.array(axes).reshape(-1)
    for ax, (title, image) in zip(axes_arr, panels.items()):
        if image.ndim == 2:
            ax.imshow(image, cmap=cmap)
        else:
            ax.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    for ax in axes_arr[len(panels) :]:
        ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
