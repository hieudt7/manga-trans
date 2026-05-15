#!/usr/bin/env python3
"""
Download and set up CCIP and anime face detector models for Koharu.

CCIP (Character Cosine Identity Proximity) is a face-embedding model trained
on anime characters from deepghs.  In recent versions of dghs-imgutils the
model is already distributed as an ONNX file on HuggingFace — no PyTorch
export step is needed.

  Input : 384 × 384 RGB, CLIP normalisation
           mean = (0.48145466, 0.4578275,  0.40821073)
           std  = (0.26862954, 0.26130258, 0.27577711)
  Output: 512-dim L2-normalised feature vector

Requirements:
    pip install dghs-imgutils huggingface_hub ultralytics

Usage:
    python tools/export_ccip.py            # both models
    python tools/export_ccip.py --ccip-only
    python tools/export_ccip.py --face-only

If you are behind a proxy that uses a self-signed certificate you can bypass
SSL verification with:
    HF_HUB_DISABLE_SSL_VERIFICATION=1 python tools/export_ccip.py
"""
import os
import platform
from pathlib import Path


def get_output_dir() -> Path:
    system = platform.system()
    if system == "Darwin":
        cache = Path.home() / "Library" / "Caches" / "koharu" / "models"
    else:
        cache = Path.home() / ".cache" / "koharu" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _ssl_no_verify():
    """Return an SSL context with certificate verification disabled."""
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _download(url: str, dest: Path, desc: str = "") -> None:
    """
    Download *url* to *dest*, bypassing SSL verification when
    HF_HUB_DISABLE_SSL_VERIFICATION=1 is set.

    Uses urllib so it works regardless of whether httpx / requests are
    installed, and the SSL patch applies reliably.
    """
    import urllib.request

    disable_ssl = os.environ.get("HF_HUB_DISABLE_SSL_VERIFICATION", "").strip() in (
        "1", "true", "yes",
    )

    label = desc or url
    print(f"  ↓ {label}")

    opener = urllib.request.build_opener()
    if disable_ssl:
        import ssl
        https_handler = urllib.request.HTTPSHandler(context=_ssl_no_verify())
        opener = urllib.request.build_opener(https_handler)

    # HuggingFace returns a redirect — follow it.
    req = urllib.request.Request(url, headers={"User-Agent": "koharu-setup/1.0"})
    with opener.open(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1 << 20  # 1 MB
        while True:
            data = resp.read(chunk)
            if not data:
                break
            f.write(data)
            downloaded += len(data)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct:3d}%  {downloaded >> 20} / {total >> 20} MB", end="", flush=True)
    if total:
        print()


def export_ccip(output_dir: Path) -> Path:
    """
    Download the CCIP feature model ONNX from HuggingFace and copy it to the
    Koharu model cache.  The file is already ONNX — no PyTorch export needed.
    """
    MODEL_NAME = "ccip-caformer-24-randaug-pruned"
    url = (
        f"https://huggingface.co/deepghs/ccip_onnx/resolve/main"
        f"/{MODEL_NAME}/model_feat.onnx"
    )
    output_path = output_dir / "ccip.onnx"

    print(f"Downloading CCIP model ({MODEL_NAME}) from deepghs/ccip_onnx …")
    _download(url, output_path, desc="model_feat.onnx")
    return output_path


def export_face_detector(output_dir: Path) -> Path:
    """
    Download the anime face detector ONNX directly from deepghs/anime_face_detection.
    The repo already ships ONNX files — no PyTorch export step needed.
    """
    onnx_url = (
        "https://huggingface.co/deepghs/anime_face_detection"
        "/resolve/main/face_detect_v1.4_s/model.onnx"
    )
    output_path = output_dir / "face_detector_anime.onnx"

    print("Downloading anime face detector from deepghs/anime_face_detection …")
    _download(onnx_url, output_path, desc="face_detect_v1.4_s/model.onnx")
    return output_path


def export_panel_detector(output_dir: Path) -> Path:
    """
    Download the manga panel detector (leoxs22/manga-panel-detector-yolo26n, YOLO26-nano)
    and export it to ONNX.

    Classes: 0 = panel (manga frame), 1 = text (speech bubble)
    Input : 640x640 RGB normalised to [0, 1]
    Recommended confidence threshold: 0.25
    """
    import shutil, tempfile

    pt_url = (
        "https://huggingface.co/leoxs22/manga-panel-detector-yolo26n"
        "/resolve/main/manga_panel_detector_fp32.pt"
    )

    print("Downloading manga panel detector from leoxs22/manga-panel-detector-yolo26n …")
    with tempfile.TemporaryDirectory() as tmp:
        pt_path = Path(tmp) / "manga_panel_detector_fp32.pt"
        _download(pt_url, pt_path, desc="manga_panel_detector_fp32.pt")

        from ultralytics import YOLO

        print("Exporting to ONNX (imgsz=640) …")
        model = YOLO(str(pt_path))
        export_path = model.export(
            format="onnx",
            imgsz=640,
            simplify=True,
            opset=17,
        )

        output_path = output_dir / "manga_panel_detector.onnx"
        shutil.copy(export_path, output_path)

    return output_path


def export_wd_tagger(output_dir: Path) -> Path:
    """
    Download WD Tagger (SmilingWolf/wd-v1-4-convnext-tagger-v2) ONNX + tags CSV.

    Used for gender + age-group classification of unknown manga/anime characters.
    Input:  [1, 448, 448, 3]  NHWC, BGR float32 [0, 255], white-padded square
    Output: [1, N_TAGS]  sigmoid probabilities for ~10k danbooru tags

    Gender is inferred from tags like 1girl/1boy; age from loli/shota/old_man/etc.
    """
    base_url = (
        "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2"
        "/resolve/main"
    )
    model_url = f"{base_url}/model.onnx"
    tags_url  = f"{base_url}/selected_tags.csv"

    model_path = output_dir / "wd_tagger.onnx"
    tags_path  = output_dir / "wd_tagger_tags.csv"

    print("Downloading WD Tagger from SmilingWolf/wd-v1-4-convnext-tagger-v2 …")
    _download(model_url, model_path, desc="model.onnx")
    _download(tags_url,  tags_path,  desc="selected_tags.csv")
    return model_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up CCIP and/or face detector models for Koharu"
    )
    parser.add_argument(
        "--ccip-only",
        action="store_true",
        help="Download only the CCIP embedding model",
    )
    parser.add_argument(
        "--face-only",
        action="store_true",
        help="Export only the anime face detector",
    )
    parser.add_argument(
        "--panel-only",
        action="store_true",
        help="Export only the manga panel detector",
    )
    parser.add_argument(
        "--gender-only",
        action="store_true",
        help="Download only the gender classifier",
    )
    args = parser.parse_args()

    output_dir = get_output_dir()
    only_one = args.ccip_only or args.face_only or args.panel_only or args.gender_only
    do_ccip   = args.ccip_only   or not only_one
    do_face   = args.face_only   or not only_one
    do_panel  = args.panel_only  or not only_one
    do_gender = args.gender_only or not only_one

    if do_ccip:
        try:
            ccip_path = export_ccip(output_dir)
            print(f"\nCCIP model saved → {ccip_path}")
        except Exception as e:
            print(f"\nFailed to download CCIP: {e}")
            print(
                "Make sure dghs-imgutils and huggingface_hub are installed:\n"
                "  pip install dghs-imgutils huggingface_hub\n"
                "If behind a proxy: HF_HUB_DISABLE_SSL_VERIFICATION=1 python tools/export_ccip.py"
            )

    if do_face:
        try:
            face_path = export_face_detector(output_dir)
            print(f"Face detector saved → {face_path}")
        except Exception as e:
            print(f"\nFailed to export face detector: {e}")
            print(
                "Make sure ultralytics and huggingface_hub are installed:\n"
                "  pip install ultralytics huggingface_hub\n"
                "If behind a proxy: HF_HUB_DISABLE_SSL_VERIFICATION=1 python tools/export_ccip.py"
            )

    if do_panel:
        try:
            panel_path = export_panel_detector(output_dir)
            print(f"Panel detector saved → {panel_path}")
        except Exception as e:
            print(f"\nFailed to export panel detector: {e}")
            print(
                "Make sure ultralytics is installed:\n"
                "  pip install ultralytics\n"
                "If behind a proxy: HF_HUB_DISABLE_SSL_VERIFICATION=1 python tools/export_ccip.py --panel-only"
            )

    if do_gender:
        try:
            wd_path = export_wd_tagger(output_dir)
            print(f"WD Tagger saved → {wd_path}")
        except Exception as e:
            print(f"\nFailed to download WD Tagger: {e}")
            print(
                "If behind a proxy: HF_HUB_DISABLE_SSL_VERIFICATION=1 python tools/export_ccip.py --gender-only"
            )

    print(
        "\nDone!  Restart the Koharu app to pick up the new models.\n"
        "Character features (add/identify) will be enabled automatically."
    )


if __name__ == "__main__":
    main()
