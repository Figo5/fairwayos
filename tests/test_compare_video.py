"""Synthetic transport tests, not perception accuracy evidence."""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('compare_video', Path(__file__).parents[1] / 'tools/compare_video.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def movie(path, count, value=70, size=(32, 24)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
    assert writer.isOpened()
    for _ in range(count):
        writer.write(np.full((size[1], size[0], 3), value, np.uint8))
    writer.release()


def test_exact_encoded_pixels_and_mapping(tmp_path):
    source, encoded = tmp_path/'source.avi', tmp_path/'encoded.avi'
    movie(source, 4)
    movie(encoded, 3, 180)
    out = tmp_path/'out'
    assert module.compare(source, encoded, 1, out) == 3
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['records'] == [{'source_frame': i+1, 'output_frame': i} for i in range(3)]
    assert len(list(out.glob('pair-*.png'))) == 2
    sheet = cv2.imread(str(out/'pair-00000-00001.png'))
    cap = cv2.VideoCapture(str(encoded))
    ok, actual = cap.read()
    cap.release()
    assert ok
    np.testing.assert_array_equal(sheet[28:52,32:64], actual)


@pytest.mark.parametrize('start,source_count,output_count,size', [
    (-1, 2, 1, (32,24)), (3, 2, 1, (32,24)),
    (1, 2, 2, (32,24)), (0, 2, 1, (48,24)),
])
def test_invalid_pair_does_not_publish_manifest(tmp_path, start, source_count, output_count, size):
    source, encoded = tmp_path/'s.avi', tmp_path/'e.avi'
    movie(source, source_count)
    movie(encoded, output_count, size=size)
    out = tmp_path/'out'
    with pytest.raises(ValueError):
        module.compare(source, encoded, start, out)
    assert not (out/'manifest.json').exists()
