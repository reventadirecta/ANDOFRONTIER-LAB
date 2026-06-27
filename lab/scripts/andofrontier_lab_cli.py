from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import runpy
import shutil
import sys
from pathlib import Path

import cv2


if hasattr(sys, "_MEIPASS"):
    sys.path.insert(0, str(Path(sys._MEIPASS)))  # type: ignore[attr-defined]
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(os.environ.get("ANDOFRONTIER_LAB_ROOT", Path.cwd())).resolve()
os.environ.setdefault("ANDOFRONTIER_LAB_ROOT", str(ROOT))


MODULE_COMMANDS = {
    "run-tracking": "scripts.run_interactive_track",
    "rebuild-from-track": "scripts.rebuild_from_track",
    "run-motion": "scripts.run_track_motion_analysis",
    "run-spectral": "scripts.run_track_spectral_analysis",
    "run-thermal": "scripts.run_track_thermal_analysis",
    "run-srv": "scripts.run_track_srv_analysis",
    "run-controls": "scripts.run_track_controls_analysis",
    "run-pca": "scripts.run_track_pca_analysis",
    "run-autoencoder": "scripts.run_track_autoencoder_analysis",
    "generate-unified-report": "scripts.generate_unified_case_report",
    "generate-reddit-template": "scripts.generate_reddit_post_template",
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _runtime_dirs(runtime_root: Path) -> dict[str, Path]:
    if (runtime_root / "scripts").exists() and (runtime_root / "uap_forensics").exists():
        lab_root = runtime_root
        logs_dir = runtime_root / "logs"
    else:
        lab_root = runtime_root / "runtime" / "lab"
        logs_dir = runtime_root / "runtime" / "logs"
    return {
        "runtime_root": runtime_root,
        "lab_root": lab_root,
        "data_root": lab_root / "data",
        "batches_dir": lab_root / "data" / "batches",
        "cases_dir": lab_root / "data" / "cases",
        "outputs_dir": lab_root / "data" / "outputs",
        "reports_dir": lab_root / "data" / "reports",
        "logs_dir": logs_dir,
    }


def _initialize_runtime(runtime_root: Path) -> dict[str, str]:
    dirs = _runtime_dirs(runtime_root.resolve())
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(path) for name, path in dirs.items()}


def _safe_slug(value: str) -> str:
    stem = Path(value).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem[:48] or "video"


def _technical_metadata(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "duration_seconds": "unknown",
            "frame_count": "unknown",
            "fps": "unknown",
            "resolution": "unknown",
            "codec": "unknown",
        }
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    cap.release()
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") or "unknown"
    return {
        "duration_seconds": round(frame_count / fps, 3) if fps > 0 and frame_count > 0 else "unknown",
        "frame_count": frame_count if frame_count > 0 else "unknown",
        "fps": round(fps, 3) if fps > 0 else "unknown",
        "resolution": {"width": width, "height": height} if width and height else "unknown",
        "codec": codec,
    }


def _read_json_or_default(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _json(payload: dict) -> int:
    print(json.dumps(payload, indent=2))
    return 0


def _run_module(module: str, args: list[str]) -> int:
    sys.argv = [module, *args]
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "module": module}, indent=2))
        return 1
    return 0


def health(_args: argparse.Namespace) -> int:
    return _json(
        {
            "ok": True,
            "app": "AndoFrontier Lab backend",
            "lab_root": str(ROOT),
            "data_dir": str(ROOT / "data"),
            "commands": ["health", "import-video", "init-runtime", "list-cases", *sorted(MODULE_COMMANDS)],
        }
    )


def init_runtime(args: argparse.Namespace) -> int:
    runtime_root = Path(args.runtime_root).resolve()
    created = _initialize_runtime(runtime_root)
    return _json(
        {
            "ok": True,
            "runtime_root": created["runtime_root"],
            "data_root": created["data_root"],
            "paths": created,
            "message": "Runtime initialized",
        }
    )


def import_video(args: argparse.Namespace) -> int:
    try:
        runtime_root = Path(args.runtime_root).resolve()
        video = Path(args.video).resolve()
        if not video.exists() or not video.is_file():
            return _json({"ok": False, "error": f"Video not found: {video}"})
        if video.suffix.lower() not in VIDEO_EXTENSIONS:
            return _json({"ok": False, "error": "Unsupported file type."})

        created = _initialize_runtime(runtime_root)
        data_root = Path(created["data_root"])
        batch_id = "local_batch"
        now = dt.datetime.now(dt.timezone.utc).astimezone()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        base_case_id = f"case_{timestamp}_{_safe_slug(video.name)}"
        case_id = base_case_id
        case_dir = data_root / "cases" / case_id
        counter = 2
        while case_dir.exists():
            case_id = f"{base_case_id}_{counter}"
            case_dir = data_root / "cases" / case_id
            counter += 1

        source_dir = case_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        copied_video = source_dir / video.name
        shutil.copy2(video, copied_video)

        technical = _technical_metadata(copied_video)
        metadata = {
            "case_id": case_id,
            "source_video": f"source/{video.name}",
            "video_path": str(copied_video),
            "original_filename": video.name,
            "created_at": now.isoformat(timespec="seconds"),
            "status": "imported",
            "tracking_status": "not_started",
            "track_based_analysis_ready": False,
            "notes": "",
            "import_mode": "copied",
            "source_quality": "user_imported_unverified",
            **technical,
        }
        _write_json(case_dir / "case_metadata.json", metadata)

        batch_dir = data_root / "batches" / batch_id
        manifest_path = batch_dir / "batch_manifest.json"
        manifest = _read_json_or_default(
            manifest_path,
            {
                "batch_id": batch_id,
                "cases": [],
                "created_by": "AndoFrontier Lab",
                "schema_version": "0.3",
            },
        )
        manifest.setdefault("batch_id", batch_id)
        manifest.setdefault("created_by", "AndoFrontier Lab")
        manifest.setdefault("schema_version", "0.3")
        manifest.setdefault("cases", [])
        case_entry = {
            "case_id": case_id,
            "video_path": str(copied_video),
            "original_filename": video.name,
            "priority": "imported",
            "quick_priority": "not_run",
            "tracking_status": "tracking_not_run",
            "review_status": "tracking_required",
            "source_quality": "user_imported_unverified",
            **technical,
        }
        manifest["cases"] = [item for item in manifest["cases"] if item.get("case_id") != case_id]
        manifest["cases"].append(case_entry)
        _write_json(manifest_path, manifest)

        return _json(
            {
                "ok": True,
                "case_id": case_id,
                "case_dir": str(case_dir),
                "source_video": str(copied_video),
                "batch_id": batch_id,
                "manifest_path": str(manifest_path),
            }
        )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc)})


def list_cases(_args: argparse.Namespace) -> int:
    cases_dir = ROOT / "data" / "cases"
    cases = sorted(path.name for path in cases_dir.iterdir() if path.is_dir()) if cases_dir.exists() else []
    return _json({"ok": True, "lab_root": str(ROOT), "cases": cases})


def dispatch(command: str, passthrough: list[str]) -> int:
    if command not in MODULE_COMMANDS:
        raise SystemExit(f"Unknown command: {command}")
    return _run_module(MODULE_COMMANDS[command], passthrough)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "-m":
        return _run_module(argv[1], argv[2:])
    if argv and argv[0] in MODULE_COMMANDS:
        return dispatch(argv[0], argv[1:])

    parser = argparse.ArgumentParser(description="AndoFrontier Lab packaged backend CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health")
    import_parser = sub.add_parser("import-video")
    import_parser.add_argument("--runtime-root", required=True)
    import_parser.add_argument("--video", required=True)
    init_parser = sub.add_parser("init-runtime")
    init_parser.add_argument("--runtime-root", required=True)
    sub.add_parser("list-cases")
    for name in MODULE_COMMANDS:
        p = sub.add_parser(name)
        p.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    if args.command == "health":
        return health(args)
    if args.command == "import-video":
        return import_video(args)
    if args.command == "init-runtime":
        return init_runtime(args)
    if args.command == "list-cases":
        return list_cases(args)
    return dispatch(args.command, getattr(args, "args", []))


if __name__ == "__main__":
    raise SystemExit(main())
