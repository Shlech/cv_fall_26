"""Воспроизводимое сравнение времени двух реализаций и сохранение графика."""

import csv
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

if __package__:
    from .morphology import dilate_native, dilate_opencv
else:
    from morphology import dilate_native, dilate_opencv


SIZES = (128, 256, 512, 1024)
REPEATS = 7
OUTPUT_DIR = Path(__file__).parent / "results"


def measure(image: np.ndarray, repeats: int = 7) -> dict:
    """Прогрев, проверка равенства и медиана независимых измерений в мс."""
    if repeats < 1:
        raise ValueError("Количество повторений должно быть положительным.")
    functions = {"native": dilate_native, "opencv": dilate_opencv}
    expected = dilate_opencv(image)
    if not np.array_equal(dilate_native(image), expected):
        raise AssertionError("Результаты реализаций не совпали.")
    samples = {name: [] for name in functions}
    for repeat in range(repeats):
        # Чередование порядка уменьшает влияние прогрева и порядка запуска.
        order = ("native", "opencv") if repeat % 2 == 0 else ("opencv", "native")
        for name in order:
            start = perf_counter_ns()
            result = functions[name](image)
            elapsed_ms = (perf_counter_ns() - start) / 1_000_000
            samples[name].append(elapsed_ms)
            # Проверка находится вне измеряемого интервала.
            if not np.array_equal(result, expected):
                raise AssertionError(f"Некорректный результат: {name}.")
    native_ms = median(samples["native"])
    opencv_ms = median(samples["opencv"])
    return {
        "native_ms": native_ms,
        "opencv_ms": opencv_ms,
        "speedup": native_ms / opencv_ms,
        "equal": True,
        "samples_ms": samples,
    }


def run_benchmark(output_dir: Path, sizes=(128, 256, 512, 1024), repeats=7) -> dict:
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("Размеры изображений должны быть положительными.")
    if repeats < 1:
        raise ValueError("Количество повторений должно быть положительным.")
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        rng = np.random.default_rng(6)
        rows = []
        for size in sizes:
            image = (rng.random((size, size)) < 0.2).astype(np.uint8) * 255
            metrics = measure(image, repeats)
            rows.append({"size": size, "pixels": size * size, **metrics})
            print(f"{size}x{size}: Python {metrics['native_ms']:.3f} ms; "
                  f"OpenCV {metrics['opencv_ms']:.3f} ms; x{metrics['speedup']:.1f}", flush=True)
        report = {
            "environment": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "opencv_threads": cv2.getNumThreads(),
            },
            "seed": 6,
            "foreground_probability": 0.2,
            "repeats": repeats,
            "rows": rows,
        }
        (output_dir / "benchmark.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fields = ("size", "pixels", "native_ms", "opencv_ms", "speedup", "equal")
        with (output_dir / "benchmark.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, label in (("native_ms", "Python"), ("opencv_ms", "OpenCV")):
            ax.plot([r["pixels"] for r in rows], [r[key] for r in rows], "o-", label=label)
        ax.set(xscale="log", yscale="log", xlabel="Число пикселей",
               ylabel="Медианное время, мс", title="Дилатация 3×3: время полного вызова")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "benchmark.png", dpi=160)
        plt.close(fig)
        return report
    finally:
        cv2.setNumThreads(previous_threads)


def main() -> None:
    run_benchmark(OUTPUT_DIR, SIZES, REPEATS)


if __name__ == "__main__":
    main()
