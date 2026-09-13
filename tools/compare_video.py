"""Make clean/encoded comparison sheets; never redraw model proposals."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def compare(source, annotated, start, output):
    if start < 0:
        raise ValueError('start must be nonnegative')
    output.mkdir(parents=True, exist_ok=False)
    a, b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(annotated))
    records, batch = [], []
    try:
        if not a.isOpened() or not b.isOpened():
            raise ValueError('both videos must open')
        for _ in range(start):
            if not a.read()[0]:
                raise ValueError('start exceeds source')
        while True:
            ok, marked = b.read()
            if not ok:
                break
            ok, clean = a.read()
            if not ok or clean.shape != marked.shape:
                raise ValueError('source exhausted or output geometry differs')
            index = len(records)
            # Native pixel size is preserved; sheets are paged, not shrunk.
            label = np.zeros((28, clean.shape[1] * 2, 3), np.uint8)
            cv2.putText(label, f'source {start + index} / output {index}: CLEAN | ENCODED',
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            tile = np.vstack([label, np.hstack([clean, marked])])
            batch.append(tile)
            records.append({'source_frame': start + index, 'output_frame': index})
            if len(batch) == 2:
                path = output / f'pair-{index - 1:05d}-{index:05d}.png'
                if not cv2.imwrite(str(path), np.vstack(batch)):
                    raise OSError(f'could not write {path}')
                batch.clear()
        if not records:
            raise ValueError('no annotated frames decoded')
        if batch:
            path = output / f'pair-{len(records)-1:05d}.png'
            if not cv2.imwrite(str(path), batch[0]):
                raise OSError(f'could not write {path}')
    finally:
        a.release()
        b.release()
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in [('source', source), ('annotated', annotated)]}
    manifest = {'hashes': hashes, 'opencv': cv2.__version__,
                'decode': 'sequential BGR; no reconstructed overlays',
                'limitation': 'mapping is caller-declared; not proof of source alignment or complete decode',
                'records': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return len(records)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('annotated', type=Path)
    p.add_argument('--start-frame', type=int, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    print(f'{compare(args.source, args.annotated, args.start_frame, args.out_dir)} encoded/source frame pairs written')
