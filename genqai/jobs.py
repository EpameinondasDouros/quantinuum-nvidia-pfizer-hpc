"""Background job runner for the workbench (Lane E).

    runner = JobRunner(max_workers=1)
    job = runner.submit(key, fn, *args, timeout=60, **kwargs)   # fn(ctx, *args, **kwargs)
    job.status       # "queued" | "running" | "done" | "failed" | "cancelled"
    job.messages     # progress messages from ctx.progress(msg)
    runner.cancel(key)

* Dedupe: submitting a key whose job is still queued or running returns that job; nothing new starts.
* Cooperative cancel and timeout: fn receives a JobContext; ctx.check() raises JobCancelled once the job is
  cancelled or past its deadline (a timeout ends as "failed" with a timeout error).
* Hard cancel: submit(..., isolate=True) runs fn(*args, **kwargs) (no ctx) in a separate spawned process,
  which cancel() or the timeout terminates. fn and its arguments must be picklable (top-level functions);
  use it for chemistry that cannot check a flag (SCF, FCI).
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

ACTIVE = ("queued", "running")


class JobCancelled(Exception):
    pass


class JobTimeout(JobCancelled):
    pass


@dataclass
class Job:
    key: str
    status: str = "queued"
    messages: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    submitted: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    timeout: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _future: Any = field(default=None, repr=False)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    @property
    def elapsed(self) -> float | None:
        if self.started is None:
            return None
        return (self.finished or time.time()) - self.started

    def wait(self, timeout: float | None = None) -> "Job":
        deadline = None if timeout is None else time.time() + timeout
        while self.active and (deadline is None or time.time() < deadline):
            time.sleep(0.01)
        return self


class JobContext:
    """Handed to fn as its first argument."""

    def __init__(self, job: Job):
        self._job = job

    def progress(self, message: str) -> None:
        self._job.messages.append(str(message))

    @property
    def cancelled(self) -> bool:
        return self._job._cancel.is_set() or self.timed_out

    @property
    def timed_out(self) -> bool:
        j = self._job
        return j.timeout is not None and j.started is not None and time.time() - j.started > j.timeout

    def check(self) -> None:
        if self.timed_out:
            raise JobTimeout(f"timed out after {self._job.timeout:g} s")
        if self._job._cancel.is_set():
            raise JobCancelled("cancelled")


def _child(conn, fn, args, kwargs):
    try:
        conn.send(("ok", fn(*args, **kwargs)))
    except BaseException as exc:  # report everything to the parent
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


class JobRunner:
    def __init__(self, max_workers: int = 1):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="genqai-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, key: str, fn: Callable, *args, timeout: float | None = None, isolate: bool = False,
               **kwargs) -> Job:
        with self._lock:
            existing = self._jobs.get(key)
            if existing is not None and existing.active:
                return existing
            job = Job(key=key, timeout=timeout)
            self._jobs[key] = job
            job._future = self._pool.submit(self._run, job, fn, args, kwargs, isolate)
        return job

    def _run(self, job: Job, fn, args, kwargs, isolate: bool) -> None:
        if job._cancel.is_set():
            job.status, job.finished = "cancelled", time.time()
            return
        job.status, job.started = "running", time.time()
        ctx = JobContext(job)
        try:
            job.result = self._run_isolated(ctx, fn, args, kwargs) if isolate else fn(ctx, *args, **kwargs)
            job.status = "done"
        except JobTimeout as exc:
            job.status, job.error = "failed", str(exc)
        except JobCancelled:
            job.status = "cancelled"
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.messages.append(traceback.format_exc(limit=3))
        finally:
            job.finished = time.time()

    @staticmethod
    def _run_isolated(ctx: JobContext, fn, args, kwargs):
        parent, child = mp.get_context("spawn").Pipe(duplex=False)
        proc = mp.get_context("spawn").Process(target=_child, args=(child, fn, args, kwargs), daemon=True)
        proc.start()
        child.close()
        ctx.progress(f"started worker process {proc.pid}")
        try:
            while not parent.poll(0.05):
                if ctx.cancelled:
                    proc.terminate()
                    ctx.check()
                if not proc.is_alive() and not parent.poll(0.05):
                    raise RuntimeError(f"worker process exited with code {proc.exitcode}")
            kind, payload = parent.recv()
        finally:
            proc.join(timeout=1)
            if proc.is_alive():
                proc.kill()
        if kind == "err":
            raise RuntimeError(payload)
        return payload

    def get(self, key: str) -> Job | None:
        return self._jobs.get(key)

    def status(self, key: str) -> str | None:
        job = self._jobs.get(key)
        return job.status if job else None

    def jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def cancel(self, key: str) -> bool:
        """Request cancellation. A queued job never starts; a running one stops at its next ctx.check()
        (or is terminated, when isolated). Returns False when there is no active job for key."""
        job = self._jobs.get(key)
        if job is None or not job.active:
            return False
        job._cancel.set()
        if job._future is not None and job._future.cancel():
            job.status, job.finished = "cancelled", time.time()
        return True

    def shutdown(self) -> None:
        for job in self._jobs.values():
            job._cancel.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
