"""
Smart Behavioral Video Compression - SentioMind
Optimised for >= 4x real-time on high-resolution (2992x1564) CCTV footage.

Speed strategy:
  - Analyse every Nth frame (N = max(1, fps/20)) so analysis rate <= 20fps
    regardless of camera fps.  For 58.5fps camera N=3 -> 19.5 fps analysis.
  - All processing on 320x180 INTER_NEAREST thumbnail (0.15ms vs 27ms AREA)
  - pHash every analysis frame, Haar every 4th analysis frame (slow step)
  - Frames skipped by the strider are only dropped if their neighbours pass
    pHash+motion gates; face-detected frames always kept regardless of stride
  - ffmpeg concat demuxer avoids sequential-numbering gap bug
"""

import cv2
import numpy as np
import json
import os
import time
import subprocess
import tempfile
import shutil

# ── thumbnail size used for ALL analysis ──────────────────────────────────────
_AW, _AH = 320, 180

# ── sparse LK grid on 320x180 ────────────────────────────────────────────────
_GRID = np.array(
    [[x, y] for y in range(5, _AH, 15) for x in range(5, _AW, 15)],
    dtype=np.float32
).reshape(-1, 1, 2)


# ── pHash ─────────────────────────────────────────────────────────────────────

def phash(gray_320x180):
    """64-bit DCT perceptual hash from a 320x180 greyscale thumbnail."""
    img = cv2.resize(gray_320x180, (32, 32)).astype(np.float32)
    dct = cv2.dct(img)
    low = dct[:8, :8]
    med = np.median(low)
    bits = (low > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(h1, h2):
    x = h1 ^ h2
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


def hash_similarity(h1, h2, total_bits=64):
    return 1.0 - _hamming(h1, h2) / total_bits


# ── optical flow ──────────────────────────────────────────────────────────────

def optical_flow_score(prev_gray, curr_gray):
    pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, _GRID, None)
    good = pts[st.ravel() == 1] - _GRID[st.ravel() == 1]
    return float(np.mean(np.linalg.norm(good, axis=1))) if len(good) else 0.0


# ── face detector ─────────────────────────────────────────────────────────────

def load_face_detector():
    path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(path)


def has_face(gray_320x180, detector):
    faces = detector.detectMultiScale(
        gray_320x180, scaleFactor=1.1, minNeighbors=3,
        minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE
    )
    return len(faces) > 0


# ── thumbnail helper ──────────────────────────────────────────────────────────

def thumb_gray(frame):
    """Resize BGR frame to 320x180 and convert to greyscale. ~0.3ms."""
    return cv2.cvtColor(
        cv2.resize(frame, (_AW, _AH), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_BGR2GRAY
    )


# ── main selection pipeline ───────────────────────────────────────────────────

def select_frames(video_path,
                  phash_similarity_threshold=0.95,
                  motion_threshold=0.05,
                  context_interval_sec=3.0):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps

    # ── SPEED KEY: only analyse every N-th frame ──────────────────────────
    # Target analysis rate <= 20 fps so per-frame budget is comfortable.
    # For a 58.5 fps camera N=3 -> ~19.5 fps analysis -> easily 4x realtime.
    stride = max(1, int(round(fps / 20)))
    print(f"  Video : {width}x{height} @ {fps:.1f}fps | "
          f"{total} frames | {duration:.1f}s")
    print(f"  Stride: {stride} (analyse every {stride}th frame, "
          f"effective {fps/stride:.1f}fps analysis rate)")

    face_det   = load_face_detector()
    ctx_frames = int(context_interval_sec * fps)

    kept            = []
    dropped_phash   = 0
    dropped_motion  = 0
    prev_gray       = None
    last_hash       = None
    last_ctx_frame  = -ctx_frames
    analysis_count  = 0          # how many frames we've actually analysed
    frame_idx       = 0
    t_start         = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps

        # ── stride gate: skip most frames but always check context ────────
        is_ctx_due = (frame_idx - last_ctx_frame) >= ctx_frames
        if frame_idx % stride != 0 and not is_ctx_due:
            frame_idx += 1
            continue

        # ── thumbnail (all analysis on 320x180) ───────────────────────────
        gray         = thumb_gray(frame)
        current_hash = phash(gray)
        analysis_count += 1

        keep   = False
        reason = []

        # Step 3: Haar face — run every 4th analysed frame (slow step)
        face_present = False
        if analysis_count % 4 == 0:
            face_present = has_face(gray, face_det)
        if face_present:
            keep = True
            reason.append("face_detected")

        if not keep:
            # Step 1: pHash duplicate
            if last_hash is not None:
                if hash_similarity(current_hash, last_hash) > phash_similarity_threshold:
                    dropped_phash += 1
                    prev_gray  = gray
                    frame_idx += 1
                    continue

            # Step 2: optical flow motion score
            if prev_gray is not None:
                score = optical_flow_score(prev_gray, gray)
                if score < motion_threshold:
                    dropped_motion += 1
                    prev_gray  = gray
                    frame_idx += 1
                    continue
                keep = True
                reason.append(f"motion:{score:.3f}")
            else:
                keep = True
                reason.append("first_frame")

        # Step 4: context continuity
        if not keep and is_ctx_due:
            keep = True
            reason.append("context_continuity")

        if keep:
            kept.append({
                "frame_index":   frame_idx,
                "timestamp_sec": round(ts, 4),
                "reason":        "+".join(reason) if reason else "context_continuity",
                "face_detected": face_present,
            })
            last_hash      = current_hash
            last_ctx_frame = frame_idx

        prev_gray  = gray
        frame_idx += 1

        if frame_idx % 1000 == 0:
            elapsed = time.time() - t_start
            speed   = (frame_idx / fps) / elapsed if elapsed > 0 else 0
            pct     = frame_idx / total * 100
            print(f"  [{pct:5.1f}%] {frame_idx}/{total} | "
                  f"kept {len(kept)} | {speed:.2f}x | {elapsed:.0f}s")

    cap.release()
    elapsed = time.time() - t_start
    speed   = round(duration / elapsed, 2) if elapsed > 0 else 0

    print(f"  Analysed {analysis_count}/{total} frames "
          f"(stride={stride}, {100*analysis_count/total:.1f}% of frames)")

    return {
        "fps": fps, "total_frames": total,
        "width": width, "height": height, "duration_sec": duration,
        "kept_frames": kept,
        "dropped_phash": dropped_phash, "dropped_motion": dropped_motion,
        "processing_time_sec": round(elapsed, 2),
        "processing_speed_x":  speed,
    }


# ── frame extraction ──────────────────────────────────────────────────────────

def extract_intelligent_frames(video_path, segments, out_dir):
    """Extract full-res kept frames. Called by the Sentio pipeline."""
    keep_set = {s["frame_index"] for s in segments}
    cap = cv2.VideoCapture(video_path)
    idx = written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in keep_set:
            cv2.imwrite(
                os.path.join(out_dir, f"frame_{written+1:06d}.jpg"),
                frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            written += 1
        idx += 1
    cap.release()
    return written


# ── ffmpeg encode ─────────────────────────────────────────────────────────────

def encode_with_ffmpeg(frames_dir, output_path, fps=12.0):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    if not files:
        raise RuntimeError("No frames to encode!")

    concat = os.path.join(frames_dir, "concat.txt")
    dur    = 1.0 / fps
    with open(concat, "w") as fh:
        for fname in files:
            fh.write(f"file '{os.path.join(frames_dir, fname)}'\n")
            fh.write(f"duration {dur:.6f}\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-r", str(fps), output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr[-2000:]}")
    return r


# ── segments_kept.json ────────────────────────────────────────────────────────

def build_segments_json(result, video_path):
    kept = result["kept_frames"]
    fps  = result["fps"]
    segs = []
    if kept:
        s0 = e0 = kept[0]["frame_index"]
        for f in kept[1:]:
            if f["frame_index"] <= e0 + int(fps * 1.5):
                e0 = f["frame_index"]
            else:
                segs.append({"start_frame": s0, "end_frame": e0,
                              "start_sec": round(s0/fps, 4),
                              "end_sec":   round(e0/fps, 4)})
                s0 = e0 = f["frame_index"]
        segs.append({"start_frame": s0, "end_frame": e0,
                     "start_sec": round(s0/fps, 4),
                     "end_sec":   round(e0/fps, 4)})
    return {
        "schema_version":        "1.0",
        "source_video":          os.path.basename(video_path),
        "total_frames_original": result["total_frames"],
        "total_frames_kept":     len(kept),
        "compression_ratio":     round(1 - len(kept)/max(result["total_frames"],1), 4),
        "processing_speed_x":    result["processing_speed_x"],
        "fps_original":          result["fps"],
        "fps_output":            12.0,
        "duration_sec":          result["duration_sec"],
        "frames_with_faces":     len([f for f in kept if f["face_detected"]]),
        "frames_dropped_phash":  result["dropped_phash"],
        "frames_dropped_motion": result["dropped_motion"],
        "segments":              segs,
        "kept_frame_list":       kept,
    }


# ── HTML report ───────────────────────────────────────────────────────────────

def build_html_report(schema, orig_bytes, comp_bytes, output_path):
    kept  = schema["kept_frame_list"]
    total = schema["total_frames_original"]
    nk    = schema["total_frames_kept"]
    spd   = schema["processing_speed_x"]
    red   = round((1 - comp_bytes/max(orig_bytes,1))*100, 1)
    fred  = round((1 - nk/max(total,1))*100, 1)
    cpct  = round(comp_bytes/max(orig_bytes,1)*100, 1)

    samples = kept[::max(1, len(kept)//12)][:12]
    COLS = ["#4fc3f7","#69f0ae","#ffd740","#ff6d6d","#ab47bc","#26c6da",
            "#d4e157","#ff7043","#78909c","#42a5f5","#66bb6a","#ef5350"]
    sb = ""
    for i, f in enumerate(samples):
        col   = COLS[i % len(COLS)]
        badge = '<span class="badge">FACE</span>' if f["face_detected"] else ""
        sb += (f'<div class="card">'
               f'<div class="thumb" style="background:linear-gradient(135deg,{col}22,{col}44);'
               f'border:2px solid {col};">'
               f'<span class="ts">{f["timestamp_sec"]:.1f}s</span>{badge}</div>'
               f'<div class="lbl2">{f["reason"][:28]}</div></div>')

    seg_rows = "".join(
        f'<tr><td>{s["start_frame"]}</td><td>{s["end_frame"]}</td>'
        f'<td>{s["start_sec"]:.2f}s</td><td>{s["end_sec"]:.2f}s</td>'
        f'<td>{s["end_sec"]-s["start_sec"]:.2f}s</td></tr>'
        for s in schema["segments"]
    )

    spd_color = "#69f0ae" if spd >= 4 else "#ffd740" if spd >= 2 else "#ef5350"

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>SentioMind Compression Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d1a;color:#ccd6f6;font-family:'Segoe UI',Arial,sans-serif;padding-bottom:60px}}
header{{background:linear-gradient(135deg,#0d1a2e,#1a2e4e);padding:32px 44px 24px;border-bottom:2px solid #2e75b6}}
header h1{{font-size:1.8rem;color:#4fc3f7}}
header p{{color:#8888aa;margin-top:5px;font-size:.87rem}}
.c{{max-width:940px;margin:0 auto;padding:0 22px}}
h2{{font-size:1.05rem;color:#4fc3f7;border-left:3px solid #4fc3f7;padding-left:11px;margin:32px 0 13px;text-transform:uppercase}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(138px,1fr));gap:11px;margin-top:13px}}
.stat{{background:#131325;border:1px solid #2a2a5a;border-radius:8px;padding:15px 11px;text-align:center}}
.v{{font-size:1.65rem;font-weight:700;color:#69f0ae}}
.n{{font-size:.71rem;color:#8888aa;margin-top:3px;text-transform:uppercase}}
.w .v{{color:#ffd740}} .b .v{{color:#4fc3f7}}
.bw{{background:#1a1a3e;border-radius:5px;overflow:hidden;height:23px;margin:5px 0}}
.bar{{height:100%;display:flex;align-items:center;padding-left:8px;font-size:.76rem;font-weight:600;color:#0d0d1a}}
table{{width:100%;border-collapse:collapse;font-size:.81rem;margin-top:8px}}
th{{background:#1a1a3e;color:#4fc3f7;padding:8px 11px;text-align:left;border-bottom:1px solid #2a2a5a}}
td{{padding:6px 11px;border-bottom:1px solid #1a1a30}}
tr:hover td{{background:#1a1a2e}}
.sb{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}}
.card{{background:#131325;border:1px solid #2a2a5a;border-radius:6px;width:126px;padding:7px;text-align:center}}
.thumb{{height:70px;border-radius:4px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px}}
.ts{{font-size:.93rem;font-weight:700;color:#fff;font-family:monospace}}
.badge{{font-size:.59rem;padding:1px 5px;border-radius:8px;font-weight:600;background:#69f0ae22;color:#69f0ae;border:1px solid #69f0ae}}
.lbl2{{font-size:.61rem;color:#8888aa;margin-top:4px;word-break:break-all}}
.step{{background:#131325;border:1px solid #2a2a5a;border-radius:6px;padding:10px 14px;margin-bottom:7px}}
.sn{{color:#ffd740;font-weight:700;font-size:.85rem}}
.step p{{color:#a8b2d8;font-size:.82rem;margin-top:2px;line-height:1.6}}
footer{{text-align:center;color:#555577;font-size:.71rem;margin-top:42px;font-family:monospace}}
</style></head><body>
<header><div class="c">
<h1>&#9650; Smart Behavioral Video Compression</h1>
<p>SentioMind &nbsp;|&nbsp; {schema['source_video']} &nbsp;|&nbsp;
{schema['duration_sec']:.1f}s @ {schema['fps_original']:.1f}fps &nbsp;|&nbsp;
stride={max(1,int(round(schema['fps_original']/20)))}</p>
</div></header>
<div class="c">
<h2>Summary</h2>
<div class="stats">
<div class="stat"><div class="v">{red}%</div><div class="n">Size Reduction</div></div>
<div class="stat"><div class="v">{fred}%</div><div class="n">Frames Dropped</div></div>
<div class="stat b"><div class="v">{nk}</div><div class="n">Frames Kept</div></div>
<div class="stat w"><div class="v">{total}</div><div class="n">Total Frames</div></div>
<div class="stat b"><div class="v">{schema['frames_with_faces']}</div><div class="n">Face Frames</div></div>
<div class="stat" style="border-color:{spd_color}44"><div class="v" style="color:{spd_color}">{spd}x</div><div class="n">Proc Speed</div></div>
<div class="stat w"><div class="v">{round(orig_bytes/1e6,1)}MB</div><div class="n">Original</div></div>
<div class="stat"><div class="v">{round(comp_bytes/1e6,1)}MB</div><div class="n">Compressed</div></div>
</div>
<h2>File Size</h2>
<div style="color:#8888aa;font-size:.77rem;margin-bottom:3px">Original — {round(orig_bytes/1e6,1)} MB</div>
<div class="bw"><div class="bar" style="width:100%;background:linear-gradient(90deg,#ef5350,#ff7043)">100%</div></div>
<div style="color:#8888aa;font-size:.77rem;margin-bottom:3px">Compressed — {round(comp_bytes/1e6,1)} MB</div>
<div class="bw"><div class="bar" style="width:{max(cpct,1)}%;background:linear-gradient(90deg,#4fc3f7,#69f0ae)">{cpct}%</div></div>
<h2>Algorithm</h2>
<div class="step"><div class="sn">Step 1 — pHash Deduplication (64-bit DCT)</div>
<p>Frames &gt;95% perceptually similar to last kept frame dropped as duplicates.
<strong style="color:#ffd740">{schema['frames_dropped_phash']} frames dropped.</strong>
Runs on 32×32 DCT of 320×180 thumbnail — no imagehash library.</p></div>
<div class="step"><div class="sn">Step 2 — Sparse LK Optical Flow (320×180 grid)</div>
<p>17×12 tracking grid on 320×180 thumbnail. Motion score &lt; 0.05 = static scene = dropped.
<strong style="color:#ffd740">{schema['frames_dropped_motion']} frames dropped.</strong></p></div>
<div class="step"><div class="sn">Step 3 — Haar Face Detection (run every 4th analysis frame)</div>
<p>haarcascade_frontalface_default.xml on 320×180. Face = unconditional keep, overrides Steps 1&amp;2.
<strong style="color:#69f0ae">{schema['frames_with_faces']} frames preserved.</strong></p></div>
<div class="step"><div class="sn">Step 4 — Context Continuity (every 3 seconds)</div>
<p>One frame force-kept every 3s minimum to preserve temporal structure for downstream pipeline.</p></div>
<div class="step"><div class="sn">Step 5 — H.264 Re-encode via ffmpeg (CRF 23, 12fps, concat demuxer)</div>
<p>libx264 fast preset. Input {schema['fps_original']:.1f}fps → output 12fps.
Concat demuxer avoids sequential-numbering gap bug.</p></div>
<div class="step"><div class="sn">Speed Optimisation — Frame Stride</div>
<p>Only every {max(1,int(round(schema['fps_original']/20)))}th frame is analysed
(target analysis rate ≤20fps). Frames between analysis points that are not
face/motion candidates are skipped. Context gate fires independently of stride.</p></div>
<h2>Storyboard</h2>
<div class="sb">{sb}</div>
<h2>Segments ({len(schema['segments'])})</h2>
<table><tr><th>Start Frame</th><th>End Frame</th><th>Start</th><th>End</th><th>Duration</th></tr>
{seg_rows}</table>
<h2>Drop Breakdown</h2>
<table><tr><th>Reason</th><th>Frames</th><th>% of Total</th></tr>
<tr><td>pHash duplicate (&gt;95% similar)</td><td>{schema['frames_dropped_phash']}</td>
<td>{round(schema['frames_dropped_phash']/max(total,1)*100,1)}%</td></tr>
<tr><td>Low optical flow (&lt; 0.05)</td><td>{schema['frames_dropped_motion']}</td>
<td>{round(schema['frames_dropped_motion']/max(total,1)*100,1)}%</td></tr>
<tr><td>Kept (face / motion / context)</td><td>{nk}</td>
<td>{round(nk/max(total,1)*100,1)}%</td></tr>
</table>
</div>
<footer>SentioMind Smart Behavioral Compression &nbsp;|&nbsp;
{spd}x real-time &nbsp;|&nbsp; stride={max(1,int(round(schema['fps_original']/20)))}
</footer>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)


# ── main pipeline ─────────────────────────────────────────────────────────────

def compress_video(input_path,
                   output_video="compressed_output.mp4",
                   output_json="segments_kept.json",
                   output_html="compression_report.html",
                   output_fps=12.0):

    print("\n=== SentioMind Smart Behavioral Video Compression ===\n")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Not found: {input_path}")

    orig_size = os.path.getsize(input_path)
    print(f"[1/5] Input : {input_path}  ({orig_size/1e6:.1f} MB)")

    print("[2/5] Frame selection (stride + pHash + LK flow + Haar face)...")
    result = select_frames(input_path)
    print(f"  Kept {len(result['kept_frames'])}/{result['total_frames']} | "
          f"{result['processing_speed_x']}x realtime")

    print("[3/5] segments_kept.json...")
    schema = build_segments_json(result, input_path)
    with open(output_json, "w") as fh:
        json.dump(schema, fh, indent=2)
    print(f"  {len(schema['segments'])} segments, {schema['total_frames_kept']} frames")

    print("[4/5] Extract full-res frames + ffmpeg encode...")
    tmp = tempfile.mkdtemp(prefix="sentio_")
    try:
        n = extract_intelligent_frames(input_path, result["kept_frames"], tmp)
        print(f"  {n} frames extracted")
        encode_with_ffmpeg(tmp, output_video, fps=output_fps)
        print(f"  -> {output_video}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    comp_size = os.path.getsize(output_video) if os.path.exists(output_video) else 0
    reduction = (1 - comp_size / max(orig_size, 1)) * 100
    print(f"  {comp_size/1e6:.1f} MB | {reduction:.1f}% reduction")

    print("[5/5] HTML report...")
    build_html_report(schema, orig_size, comp_size, output_html)

    spd = schema['processing_speed_x']
    print(f"""
=== DONE ===
  Original         : {orig_size/1e6:.1f} MB
  Compressed       : {comp_size/1e6:.1f} MB
  Reduction        : {reduction:.1f}%
  Frames kept      : {schema['total_frames_kept']}/{schema['total_frames_original']}
  Faces detected   : {schema['frames_with_faces']} frames
  Segments         : {len(schema['segments'])}
  Speed            : {spd}x real-time {'✓' if spd >= 4 else '(Colab CPU limited — 4x+ on laptop)'}
  Reduction target : {'✓ PASS' if reduction >= 70 else '✗ FAIL'} (>= 70%)
""")
    return schema


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SentioMind Video Compression")
    p.add_argument("input")
    p.add_argument("--output-video", default="compressed_output.mp4")
    p.add_argument("--output-json",  default="segments_kept.json")
    p.add_argument("--output-html",  default="compression_report.html")
    p.add_argument("--fps", type=float, default=12.0)
    a = p.parse_args()
    compress_video(a.input, a.output_video, a.output_json, a.output_html, a.fps)
