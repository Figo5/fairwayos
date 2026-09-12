"""Bounded local job store: progress, error, cancel, cleanup.

State lives in memory with a private per-job working directory under the store
root. Nothing is shared between jobs and nothing is written outside the root.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


class JobState:
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_SEEDS = "awaiting_seeds"   # unseeded work done; assisted work pending
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    TERMINAL = (DONE, FAILED, CANCELLED)


@dataclass
class Job:
    id: str
    filename: str
    state: str = JobState.QUEUED
    progress: float = 0.0
    error: str = ""
    created: float = field(default_factory=time.time)
    result: Optional[dict] = None
    awaiting: list = field(default_factory=list)   # targets still needing a seed
    media_retention: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# Media has to outlive the first phase: a reviewer cannot pick a seed on a clip
# that was deleted the moment the unseeded pass finished. Retention is therefore
# bounded by a deadline rather than by job completion, and is still explicit,
# still confined to this root, and still droppable on demand.
DEFAULT_MEDIA_RETENTION_SECONDS = 3600


class JobStore:
    def __init__(self, root: str,
                 media_retention_seconds: float = DEFAULT_MEDIA_RETENTION_SECONDS):
        self.media_retention_seconds = float(media_retention_seconds)
        self.root = os.path.realpath(root)
        os.makedirs(self.root, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._cancel: Dict[str, bool] = {}
        self._sources: Dict[str, object] = {}
        self._seeds: Dict[str, object] = {}
        self._procs: Dict[str, object] = {}
        self._media_deadline: Dict[str, float] = {}
        self._media_dropped: Dict[str, str] = {}
        self._media_revoked: Dict[str, bool] = {}
        self.import_dir = os.path.join(self.root, 'import')
        os.makedirs(self.import_dir, exist_ok=True)
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        j = Job(id=uuid.uuid4().hex[:16], filename=os.path.basename(filename))
        with self._lock:
            self._jobs[j.id] = j
            self._cancel[j.id] = False
        return j

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"unknown job {job_id}")
            return self._jobs[job_id]

    def list(self):
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: -j.created)

    def workdir(self, job_id: str) -> str:
        self.get(job_id)
        return os.path.join(self.root, job_id)

    def start(self, job_id: str):
        with self._lock:
            self._jobs[job_id].state = JobState.RUNNING

    def await_seeds(self, job_id: str, targets, result: dict):
        """Phase 1 finished. Hold here so a seed can actually be supplied."""
        with self._lock:
            j = self._jobs[job_id]
            if j.state in JobState.TERMINAL:
                return
            j.state, j.awaiting, j.result = (JobState.AWAITING_SEEDS,
                                             list(targets), result)
            j.progress = 0.5

    def progress(self, job_id: str, value: float):
        with self._lock:
            self._jobs[job_id].progress = max(0.0, min(1.0, float(value)))

    def finish(self, job_id: str, result: dict):
        with self._lock:
            j = self._jobs[job_id]
            j.state, j.progress, j.result = JobState.DONE, 1.0, result
            j.awaiting = []

    def fail(self, job_id: str, error: str):
        with self._lock:
            j = self._jobs[job_id]
            j.state, j.error = JobState.FAILED, str(error)

    def cancel(self, job_id: str):
        with self._lock:
            self._cancel[job_id] = True
            j = self._jobs[job_id]
            if j.state not in JobState.TERMINAL:
                j.state = JobState.CANCELLED

    def cancelled(self, job_id: str) -> bool:
        with self._lock:
            return self._cancel.get(job_id, False)

    def attach_source(self, job_id: str, video):
        with self._lock:
            self._sources[job_id] = video
            self._media_deadline[job_id] = (time.time()
                                            + self.media_retention_seconds)

    def source(self, job_id: str):
        with self._lock:
            return self._sources.get(job_id)

    def attach_seeds(self, job_id: str, bundle):
        with self._lock:
            self._seeds[job_id] = bundle

    def seeds(self, job_id: str):
        with self._lock:
            return self._seeds.get(job_id)

    # --- finding 4: track the active child so cancel can actually stop it
    def set_proc(self, job_id: str, proc):
        with self._lock:
            self._procs[job_id] = proc

    def clear_proc(self, job_id: str):
        with self._lock:
            self._procs.pop(job_id, None)

    def kill_proc(self, job_id: str):
        with self._lock:
            p = self._procs.get(job_id)
        if p is None:
            return False
        try:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
            return True
        except Exception:
            return False

    # --- finding 5: explicit, bounded media retention
    def media_state(self, job_id: str) -> dict:
        """Retention state as served to the client. Never a bare boolean."""
        with self._lock:
            v = self._sources.get(job_id)
            deadline = self._media_deadline.get(job_id)
            dropped = self._media_dropped.get(job_id)
        with self._lock:
            revoked = self._media_revoked.get(job_id, False)
        path = getattr(v, "path", None) if v else None
        retained = bool(path and os.path.exists(path)) and not revoked
        return {"retained": retained,
                "retention_seconds": self.media_retention_seconds,
                "expires_at": deadline,
                "reason": dropped or ("retained for review until the deadline"
                                      if retained else "no media")}

    def enforce_retention(self, job_id: str) -> dict:
        """Drop media whose bounded window has passed. Called on every access,
        so waiting for a seed can never keep media indefinitely."""
        with self._lock:
            deadline = self._media_deadline.get(job_id)
        if deadline is not None and time.time() >= deadline:
            self.revoke_media(job_id, reason="retention window expired")
        return self.media_state(job_id)

    def revoke_media(self, job_id: str, reason: str = "deleted on request") -> dict:
        """End review access to this job's media, and DELETE the bytes when they
        are ours to delete.

        A multipart upload lives under root/<job id> and is removed. A file the
        operator asked us to analyse in place, in the import directory, is the
        operator's own file: access is revoked but the file is never deleted --
        that containment rule from the security review is unchanged.
        """
        with self._lock:
            self._media_revoked[job_id] = True
            self._media_dropped[job_id] = reason
        deleted = self.drop_source_media(job_id, reason=reason)
        st = self.media_state(job_id)
        st["bytes_deleted"] = deleted
        if not deleted:
            st["reason"] = (f"{reason}; the source is the operator's own file in "
                            f"the import directory, so access was revoked but the "
                            f"file was NOT deleted")
        return st

    def drop_source_media(self, job_id: str, reason: str = "deleted on request") -> bool:
        """Delete the uploaded media for a job, keeping metadata."""
        with self._lock:
            v = self._sources.get(job_id)
        if v is None:
            return False
        path = getattr(v, "path", None)
        if not path:
            return False
        real = os.path.realpath(path)
        # only ever delete inside our own root; never a user's import file
        upload_area = os.path.realpath(os.path.join(self.root, job_id))
        if not real.startswith(upload_area + os.sep):
            return False
        try:
            os.remove(real)
            with self._lock:
                self._media_dropped[job_id] = reason
            return True
        except FileNotFoundError:
            with self._lock:
                self._media_dropped.setdefault(job_id, reason)
            return False
        except OSError:
            return False

    def cleanup(self, job_id: str):
        with self._lock:
            self._media_dropped.setdefault(job_id, "job cleaned up")
        wd = os.path.join(self.root, job_id)
        if os.path.realpath(wd).startswith(self.root + os.sep) and os.path.isdir(wd):
            shutil.rmtree(wd, ignore_errors=True)
