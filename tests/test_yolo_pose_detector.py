import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ghostcaddie.video.observations import VideoObservations
from ghostcaddie.video.yolo_pose_detector import create_detector


class YoloPoseDetectorTests(unittest.TestCase):
    def test_detector_leaves_anchor_unavailable_without_confident_ankles(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "frame_000001.jpg"
            frame.write_bytes(b"placeholder")
            fake_result = _FakeResult()
            fake_result.keypoints = _FakeKeypointsWithoutAnkles()
            with patch("ghostcaddie.video.yolo_pose_detector.cv2") as cv2, \
                    patch("ghostcaddie.video.yolo_pose_detector.YOLO") as factory:
                cv2.imread.return_value = _FakeImage()
                factory.return_value.return_value = [fake_result]
                result = create_detector().detect([str(frame)])

        self.assertIsNone(result.items[0].golfer.anchor)

    def test_detector_returns_validated_pixel_observations_with_explicit_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "frame_000001.jpg"
            frame.write_bytes(b"placeholder")
            fake_result = _FakeResult()
            with patch("ghostcaddie.video.yolo_pose_detector.cv2") as cv2, \
                    patch("ghostcaddie.video.yolo_pose_detector.YOLO") as factory:
                cv2.imread.return_value = _FakeImage()
                factory.return_value.return_value = [fake_result]
                result = create_detector().detect([str(frame)])

        self.assertIsInstance(result, VideoObservations)
        self.assertEqual(result.schema_version, "video-observations.v1")
        self.assertEqual([item.frame_index for item in result.items], [0])
        self.assertEqual([item.timestamp_seconds for item in result.items], [0.0])
        self.assertIsNotNone(result.items[0].golfer.bbox)
        self.assertIsNone(result.items[0].ball)
        self.assertIsNone(result.items[0].clubhead)
        self.assertIn("ball_missing", result.items[0].warnings)
        self.assertNotIn("clubhead_missing", result.items[0].warnings)


class _FakeTensor:
    def __init__(self, value):
        self._value = value

    def cpu(self):
        return self

    def numpy(self):
        return self._value

    def tolist(self):
        return self._value

    def __getitem__(self, index):
        return _FakeTensor(self._value[index])

    def __float__(self):
        return float(self._value)


class _FakeImage:
    shape = (240, 320, 3)


class _FakeBoxes:
    cls = _FakeTensor([0])
    conf = _FakeTensor([0.91])
    xyxy = _FakeTensor([[10.0, 20.0, 110.0, 220.0]])

    def __len__(self):
        return 1


class _FakeKeypoints:
    xy = _FakeTensor([[[60.0, 30.0]] * 17])
    conf = _FakeTensor([[0.9] * 17])


class _FakeKeypointsWithoutAnkles:
    xy = _FakeTensor([[[60.0, 30.0]] * 17])
    conf = _FakeTensor([[0.9] * 15 + [0.1, 0.1]])


class _FakeResult:
    boxes = _FakeBoxes()
    keypoints = _FakeKeypoints()


if __name__ == "__main__":
    unittest.main()
