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

    def to_dict(self) -> dict:
        return asdict(self)


class JobStore:
    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        os.makedirs(self.root, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._cancel: Dict[str, bool] = {}
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

    def progress(self, job_id: str, value: float):
        with self._lock:
            self._jobs[job_id].progress = max(0.0, min(1.0, float(value)))

    def finish(self, job_id: str, result: dict):
        with self._lock:
            j = self._jobs[job_id]
            j.state, j.progress, j.result = JobState.DONE, 1.0, result

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

    def cleanup(self, job_id: str):
        wd = os.path.join(self.root, job_id)
        if os.path.realpath(wd).startswith(self.root + os.sep) and os.path.isdir(wd):
            shutil.rmtree(wd, ignore_errors=True)
