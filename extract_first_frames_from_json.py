#!/usr/bin/env python3
# save as extract_first_frames_from_json.py
import argparse, json, subprocess, shutil, sys, hashlib
from pathlib import Path
from collections import defaultdict

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit("Error: ffmpeg not found in PATH.")

def load_jobs(p: Path):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            sys.exit("Error: JSON root must be a list.")
        return data
    except Exception as e:
        sys.exit(f"Error reading JSON: {e}")

def short_hash(s: str, n: int = 8) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def unique_name(stem: str, src: str, seen):
    k = stem.lower()
    seen[k] += 1
    return f"{stem}.png" if seen[k] == 1 else f"{stem}__{short_hash(src)}.png"

def extract_first_frame(src: str, dst: str, fmt: str, jpg_quality: int):
    # Build ffmpeg command depending on format
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-frames:v", "1"]
    if fmt == "png":
        # PNG is lossless: closest to “same quality”
        cmd += [dst]
    else:
        # JPG path (still lossy): use highest quality by default (-q:v 1)
        cmd += ["-q:v", str(jpg_quality), dst]
    return subprocess.run(cmd).returncode == 0

def main():
    ap = argparse.ArgumentParser(description="Extract first frame from each input_video in jobs.json")
    ap.add_argument("--json", required=True, help="Path to jobs.json (list of objects with 'input_video').")
    ap.add_argument("--out", required=True, help="Output folder.")
    ap.add_argument("--format", choices=["png", "jpg"], default="png",
                    help="Output image format. Default: png (lossless).")
    ap.add_argument("--jpg-quality", type=int, default=1,
                    help="JPG quality (1=best..31=worst). Used only when --format=jpg. Default: 1")
    ap.add_argument("--skip-missing", action="store_true", help="Skip missing files instead of erroring.")
    args = ap.parse_args()

    ensure_ffmpeg()
    jobs = load_jobs(Path(args.json))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    seen = defaultdict(int)
    ok = fail = skipped = 0

    for i, item in enumerate(jobs, 1):
        src = (item.get("input_video") or "").strip()
        if not src:
            print(f"[{i}] WARN: empty input_video; skip"); skipped += 1; continue
        sp = Path(src)
        if not sp.is_file():
            if args.skip_missing:
                print(f"[{i}] SKIP (missing): {src}"); skipped += 1; continue
            else:
                print(f"[{i}] ERROR missing: {src}"); fail += 1; continue

        stem = sp.stem
        # Build output filename with collision-safe naming
        if args.format == "png":
            out_name = unique_name(stem, str(sp), seen)
        else:
            # JPG variant naming
            key = stem.lower(); seen[key] += 1
            out_name = f"{stem}.jpg" if seen[key] == 1 else f"{stem}__{short_hash(str(sp))}.jpg"

        dst = out_dir / out_name

        if extract_first_frame(str(sp), str(dst), args.format, args.jpg_quality):
            print(f"[{i}] OK  -> {dst}"); ok += 1
        else:
            print(f"[{i}] FAIL-> {src}"); fail += 1

    print(f"\nDone. OK={ok}, FAIL={fail}, SKIPPED={skipped}. Out: {out_dir}")

if __name__ == "__main__":
    main()
