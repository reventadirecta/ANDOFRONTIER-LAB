import importlib
import shutil
import sys


REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "pandas": "pandas",
    "tqdm": "tqdm",
}


def main() -> int:
    print("UAP forensic lab environment check")
    missing = []
    for module, package in REQUIRED_MODULES.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "installed")
            print(f"[ok] {package}: {version}")
        except Exception as exc:
            print(f"[missing] {package}: {exc}")
            missing.append(package)
    for binary in ("ffmpeg", "ffprobe"):
        found = shutil.which(binary)
        print(f"[{'ok' if found else 'warn'}] {binary}: {found or 'not found on PATH'}")
    if missing:
        print("\nInstall dependencies with: pip install -r requirements.txt")
        return 1
    print("\nEnvironment is ready for local analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
