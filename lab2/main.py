"""CLI: python main.py data/mona-lisa.avi [--output-dir results]."""

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np

from tracker import Detection, PlanarTracker, annotate


def process_video(source, output_dir, label="Mona Lisa"):
    source, output_dir = Path(source), Path(output_dir)
    output_path = output_dir / (source.stem + "-tracked.avi")
    if source.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ.")
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        ok, first = capture.read()
        if not ok:
            raise ValueError(f"Cannot read first frame: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video FPS: {fps}")
        cv2.setRNGSeed(42)
        tracker = PlanarTracker(first)
        output_dir.mkdir(parents=True, exist_ok=True)
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        if not writer.isOpened():
            raise ValueError(f"Cannot create video: {output_path}")
        rows = []
        frame = first
        index = 0
        while True:
            start = time.perf_counter()
            detection = (Detection(tracker.corners.copy()) if index == 0
                         else tracker.detect(frame))
            elapsed = time.perf_counter() - start
            canvas = annotate(frame, detection, label)
            writer.write(canvas)
            if index in (0, 100, 200, 300, 400, 500):
                cv2.imwrite(str(output_dir / f"{source.stem}-{index:04d}.jpg"), canvas)
            row = {
                "frame": index, "time_s": index / fps,
                "status": "reference" if index == 0 else (
                    "detected" if detection.corners is not None else "lost"),
                "matches": detection.matches, "inliers": detection.inliers,
                "method": "reference" if index == 0 else detection.method,
                "median_reprojection_error_px": detection.error,
                "processing_ms": elapsed * 1000,
            }
            for corner in range(4):
                for axis, name in enumerate(("x", "y")):
                    row[f"{name}{corner}"] = (float(detection.corners[corner, axis])
                                              if detection.corners is not None else None)
            rows.append(row)
            index += 1
            ok, frame = capture.read()
            if not ok:
                break
        with (output_dir / f"{source.stem}.csv").open("w", newline="", encoding="utf-8") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        measured = rows[1:]
        found = [row for row in measured if row["status"] == "detected"]
        seconds = sum(row["processing_ms"] for row in measured) / 1000
        summary = {
            "input": str(source), "output": str(output_path), "frames": len(rows),
            "fps": fps, "size": [w, h], "reference_keypoints": len(tracker.points),
            "evaluated_frames": len(measured), "detected_frames": len(found),
            "lost_frames": len(measured) - len(found),
            "sift_frames": sum(row["method"] == "sift" for row in found),
            "flow_frames": sum(row["method"] == "flow" for row in found),
            "detection_rate": len(found) / len(measured) if measured else None,
            "processing_fps": len(measured) / seconds if seconds else None,
            "median_reprojection_error_px": float(np.median([
                row["median_reprojection_error_px"] for row in found])) if found else None,
            "opencv": cv2.__version__, "numpy": np.__version__,
        }
        (output_dir / f"{source.stem}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description="Track a planar object from the first video frame.")
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--label", default="Mona Lisa", help="ASCII label drawn above the object")
    args = parser.parse_args()
    # Avoid collisions in batch mode before producing any output.
    if len({path.stem for path in args.videos}) != len(args.videos):
        parser.error("Input videos must have different filename stems.")
    for path in args.videos:
        try:
            summary = process_video(path, args.output_dir, args.label)
        except (ValueError, OSError, cv2.error) as error:
            parser.exit(1, f"Error: {error}\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
