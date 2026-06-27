from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _make_video(path: Path, color: tuple[int, int, int]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create fixture video: {path}")
    for idx in range(5):
        frame = np.full((48, 64, 3), color, dtype=np.uint8)
        cv2.putText(frame, str(idx), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def _parse_json(stdout: str) -> dict:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"CLI did not return JSON: {stdout}")
    return json.loads(stdout[start : end + 1])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="andofrontier_batch_import_") as tmp:
        root = Path(tmp)
        runtime = root / "runtime"
        videos = [
            root / "fixture_A.mp4",
            root / "fixture_B.mp4",
            root / "fixture_C.mp4",
        ]
        _make_video(videos[0], (40, 30, 20))
        _make_video(videos[1], (20, 80, 30))
        _make_video(videos[2], (20, 30, 100))

        cli = Path(__file__).resolve().parent / "andofrontier_lab_cli.py"
        command = [sys.executable, str(cli), "import-video", "--runtime-root", str(runtime)]
        for video in videos:
            command.extend(["--video", str(video)])
        result = subprocess.run(command, cwd=str(cli.parents[1]), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)
        payload = _parse_json(result.stdout)
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "import failed"))

        imported = payload.get("imported_cases") or []
        case_ids = [item["case_id"] for item in imported]
        if len(imported) != 3 or len(set(case_ids)) != 3:
            raise AssertionError(f"Expected 3 unique cases, got {case_ids}")

        data_root = runtime / "runtime" / "lab" / "data"
        for item in imported:
            case_dir = Path(item["case_dir"])
            metadata_path = case_dir / "case_metadata.json"
            if not metadata_path.exists():
                raise AssertionError(f"Missing metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["case_id"] != item["case_id"]:
                raise AssertionError(f"Metadata case mismatch: {metadata_path}")
            source_files = [path for path in (case_dir / "source").iterdir() if path.is_file()]
            if len(source_files) != 1:
                raise AssertionError(f"Expected exactly 1 source video in {case_dir}, got {len(source_files)}")
            if Path(metadata["video_path"]).resolve() != source_files[0].resolve():
                raise AssertionError(f"Metadata points to wrong source video: {metadata_path}")
            for isolated_root in ("outputs", "reports"):
                mixed = data_root / isolated_root / item["case_id"]
                if mixed.exists() and any(mixed.iterdir()):
                    raise AssertionError(f"Unexpected generated {isolated_root} during import: {mixed}")

        manifest = json.loads((data_root / "batches" / "local_batch" / "batch_manifest.json").read_text(encoding="utf-8"))
        manifest_ids = [item["case_id"] for item in manifest.get("cases", [])]
        if sorted(manifest_ids) != sorted(case_ids):
            raise AssertionError(f"Manifest mismatch: {manifest_ids} != {case_ids}")

        print(json.dumps({"ok": True, "case_ids": case_ids, "runtime": str(runtime)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
