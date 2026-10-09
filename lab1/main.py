import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

if __package__:
    from .benchmark import measure
    from .morphology import binarize, dilate_native, dilate_opencv
else:
    from benchmark import measure
    from morphology import binarize, dilate_native, dilate_opencv


INPUT_PATH = None  # Для своего изображения: Path("photo.jpg").
THRESHOLD = 127
INVERT = False
REPEATS = 7
OUTPUT_DIR = Path(__file__).parent / "results"


def demo_image() -> np.ndarray:
    """Авторский пример: тонкие линии, разрывы, отверстия и объекты у границ."""
    image = np.zeros((160, 240), dtype=np.uint8)
    cv2.putText(image, "CV 6", (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 1)
    cv2.rectangle(image, (20, 75), (70, 125), 255, 1)
    cv2.circle(image, (110, 100), 25, 255, 1)
    cv2.line(image, (152, 80), (152, 125), 255, 1)
    cv2.line(image, (155, 80), (155, 125), 255, 1)
    image[100, 152] = 0
    cv2.rectangle(image, (190, 90), (225, 125), 255, -1)
    image[106, 205] = 0
    image[0, 0] = image[-1, -1] = 255
    return image


def read_gray(path: Path) -> np.ndarray:
    # imdecode/fromfile поддерживают кириллицу в путях на Windows.
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    encoded.tofile(path)


def main() -> None:
    gray = read_gray(INPUT_PATH) if INPUT_PATH else demo_image()
    binary = binarize(gray, THRESHOLD, INVERT)
    native = dilate_native(binary)
    opencv = dilate_opencv(binary)
    difference = cv2.absdiff(native, opencv)
    metrics = measure(binary, REPEATS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = (gray, binary, native, opencv, difference)
    names = ("input", "binary", "native", "opencv", "difference")
    for name, image in zip(names, images):
        save_png(OUTPUT_DIR / f"{name}.png", image)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    titles = ("Исходное", "Бинаризация", "Python, 3×3", "OpenCV, 3×3", "Разность")
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=160)
    plt.close(fig)
    details = {
        "input": str(INPUT_PATH) if INPUT_PATH else "synthetic demo",
        "shape": list(binary.shape),
        "threshold": THRESHOLD,
        "invert": INVERT,
        "repeats": REPEATS,
        "different_pixels": int(np.count_nonzero(difference)),
        "white_pixels_before": int(np.count_nonzero(binary)),
        "white_pixels_after": int(np.count_nonzero(native)),
        **metrics,
    }
    (OUTPUT_DIR / "processing.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Different pixels: {details['different_pixels']}")
    print(f"Python: {metrics['native_ms']:.3f} ms; OpenCV: {metrics['opencv_ms']:.3f} ms")
    print(f"Speedup: x{metrics['speedup']:.1f}")
    print(f"Results: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
