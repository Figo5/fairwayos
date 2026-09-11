"""Run one adapter inside its own interpreter, as a subprocess.

Invoked as:  <interpreter> -m ghostcaddie.upload.worker <target> <request.json>
Writes a JSON result to stdout. The serving process never imports torch or any
model runtime; it only reads this JSON back.
"""
from __future__ import annotations

import json
import sys


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print(json.dumps({"ok": False, "error": "usage: worker <target> <request.json>"}))
        return 2
    target, req_path = argv
    try:
        req = json.load(open(req_path))
        import cv2
        from ghostcaddie.upload.adapters import all_adapters, AdapterUnavailable
        ad = all_adapters()[target]
        cap = cv2.VideoCapture(req["video"])
        frames, i = [], 0
        step = max(1, int(req.get("sampling_step", 1)))
        limit = int(req.get("max_frames", 40))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append((i, fr))
                if len(frames) >= limit:
                    break
            i += 1
        cap.release()
        recs = ad.run_frames(frames, source_sha256=req["source_sha256"])
        print(json.dumps({"ok": True, "target": target, "records": recs,
                          "frames_analysed": len(recs), "sampling_step": step}))
        return 0
    except Exception as e:                      # includes AdapterUnavailable
        print(json.dumps({"ok": False, "target": target,
                          "error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
