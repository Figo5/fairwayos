"""Active automatic work survives source expiry; revocation still deletes."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ghostcaddie.upload.jobs import JobStore


class ProcessingRetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = JobStore(self.tmp.name, media_retention_seconds=5)
        self.job = self.store.create('clip.mp4')
        directory = Path(self.store.workdir(self.job.id));directory.mkdir()
        self.path = directory / 'clip.mp4';self.path.write_bytes(b'owned upload')
        with patch('ghostcaddie.upload.jobs.time.time', return_value=100):
            self.store.attach_source(self.job.id, SimpleNamespace(path=str(self.path)))

    def test_active_worker_keeps_source_and_ready_result_gets_a_bounded_window(self):
        with self.store.processing_source(self.job.id):
            self.store.start(self.job.id)
            with patch('ghostcaddie.upload.jobs.time.time', return_value=110):
                self.assertTrue(self.store.enforce_retention(self.job.id)['retained'])
                result = {'media_retention':self.store.media_state(self.job.id)}
                self.store.finish(self.job.id, result)
                self.assertEqual(result['media_retention']['expires_at'], 115)
        with patch('ghostcaddie.upload.jobs.time.time', return_value=114):
            self.assertTrue(self.store.enforce_retention(self.job.id)['retained'])
        with patch('ghostcaddie.upload.jobs.time.time', return_value=116):
            self.assertFalse(self.store.enforce_retention(self.job.id)['retained'])
        self.assertFalse(self.path.exists())

    def test_explicit_revocation_remains_immediate_and_finish_cannot_restore_access(self):
        with self.store.processing_source(self.job.id):
            self.assertTrue(self.store.revoke_media(self.job.id)['bytes_deleted'])
            with patch('ghostcaddie.upload.jobs.time.time', return_value=110):
                self.store.finish(self.job.id, {})
        self.assertFalse(self.store.media_state(self.job.id)['retained'])
        self.assertFalse(self.path.exists())

    def test_exception_releases_protection_and_old_nonprocessing_policy_is_unchanged(self):
        with self.assertRaises(RuntimeError):
            with self.store.processing_source(self.job.id):
                raise RuntimeError('worker failed')
        with patch('ghostcaddie.upload.jobs.time.time', return_value=110):
            self.assertFalse(self.store.enforce_retention(self.job.id)['retained'])
        self.assertFalse(self.path.exists())

    def test_cancellation_does_not_start_a_new_download_window(self):
        with self.store.processing_source(self.job.id):
            self.store.cancel(self.job.id)
            with patch('ghostcaddie.upload.jobs.time.time', return_value=110):
                self.store.finish(self.job.id, {})
        with patch('ghostcaddie.upload.jobs.time.time', return_value=110):
            self.assertFalse(self.store.enforce_retention(self.job.id)['retained'])
