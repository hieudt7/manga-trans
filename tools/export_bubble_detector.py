#!/usr/bin/env python3
"""
Export YOLOv11 best.pt to ONNX for use by the Rust app.
Requirements: pip install ultralytics

Usage:
    python tools/export_bubble_detector.py --pt /path/to/best.pt
"""
import argparse
import shutil
from pathlib import Path


def get_output_path() -> Path:
    import platform
    if platform.system() == "Darwin":
        cache = Path.home() / "Library" / "Caches" / "koharu" / "models"
    else:
        cache = Path.home() / ".cache" / "koharu" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / "bubble_detector.onnx"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="Path to best.pt downloaded from HuggingFace")
    args = parser.parse_args()

    pt_path = Path(args.pt)
    if not pt_path.exists():
        print(f"Error: {pt_path} not found")
        return

    from ultralytics import YOLO

    print(f"Loading model from {pt_path}...")
    model = YOLO(str(pt_path))

    print("Exporting to ONNX (imgsz=640)...")
    export_path = model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        opset=17,
    )
    print(f"Exported to: {export_path}")

    output = get_output_path()
    shutil.copy(export_path, output)
    print(f"\nDone! Model saved to: {output}")
    print("You can now run the app — it will load the model automatically.")


if __name__ == "__main__":
    main()
