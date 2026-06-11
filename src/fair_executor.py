"""Fair shared executor for cross-equation candidate evaluation."""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from collections import Counter, deque
from concurrent.futures import Future
from typing import Any, Callable, Deque, Dict, Optional, Tuple


QueuedTask = Tuple[int, Future, Callable[..., Any], tuple, dict, str, float, int, str]
RunningTask = Tuple[str, str, float, float, int]

_WORKER_EXEC_STARTED = 0
_WORKER_EXEC_FINISHED = 1
_WORKER_PUT_STARTED = 2
_WORKER_PUT_FINISHED = 3
_WORKER_PUT_FAILED = 4
_WORKER_COUNTER_COUNT = 5


def _worker_counter_offset(worker_idx: int, idx: int) -> int:
    return worker_idx * _WORKER_COUNTER_COUNT + idx


def _inc_worker_counter(counters: Any, worker_idx: int, idx: int) -> None:
    try:
        counters[_worker_counter_offset(worker_idx, idx)] += 1
    except Exception:
        pass


def _worker_counter_sum(counters: Any, worker_count: int, idx: int) -> int:
    try:
        return sum(
            int(counters[_worker_counter_offset(worker_idx, idx)])
            for worker_idx in range(worker_count)
        )
    except Exception:
        return 0


def _process_worker(
    worker_idx: int,
    task_queue: Any,
    result_writer: Any,
    counters: Any,
) -> None:
    """Run FairEvalExecutor tasks in a single-threaded process.

    The eval/degen task functions may create their own short-lived subprocesses
    for hard timeouts.  Running this loop in a process, rather than in a parent
    thread, avoids fork-after-threads deadlocks in NumPy/SciPy/SymPy/OpenBLAS.
    """
    while True:
        task = task_queue.get()
        if task is None:
            return
        task_id, fn, args, kwargs = task
        _inc_worker_counter(counters, worker_idx, _WORKER_EXEC_STARTED)
        try:
            payload = fn(*args, **kwargs)
            ok = True
        except BaseException as exc:
            payload = repr(exc)
            ok = False
        finally:
            _inc_worker_counter(counters, worker_idx, _WORKER_EXEC_FINISHED)
        _inc_worker_counter(counters, worker_idx, _WORKER_PUT_STARTED)
        try:
            result_writer.send((task_id, ok, payload))
        except BaseException:
            _inc_worker_counter(counters, worker_idx, _WORKER_PUT_FAILED)
            raise
        else:
            _inc_worker_counter(counters, worker_idx, _WORKER_PUT_FINISHED)


class FairEvalExecutor:
    """A process-backed shared executor with one FIFO queue per owner.

    ``ThreadPoolExecutor`` uses one global FIFO queue, so the first equation to
    submit hundreds of candidates can monopolize the worker backlog.  This
    executor keeps per-equation queues and dispatches work from the owner with
    the least remaining in-flight work, so equations near the end of a step can
    finish and start producing the next step's candidates sooner.

    The workers are processes, not threads.  Candidate evaluation uses SymPy,
    SciPy/NumPy, and timeout subprocesses; forking those helpers from a large
    multithreaded parent can inherit locked pthread/BLAS/malloc state and leave
    children asleep in futex waits.  A single-threaded process worker is a safer
    place to start the per-task timeout helper.

    Tasks are sent through one input queue per worker, and results return
    through one pipe and one reader thread per worker.  This avoids global
    input/output queue locks turning one slow handoff into a cluster-wide stall.
    """

    def __init__(
        self,
        max_workers: int,
        owner_max_inflight: int = 32,
        *,
        start_method: str = "spawn",
        dynamic_workers: bool = True,
        initial_workers: int = 1,
        cpu_grow_threshold: float = 85.0,
        cpu_shrink_threshold: float = 90.0,
        cpu_check_interval: float = 0.1,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers
        self.owner_max_inflight = max(0, int(owner_max_inflight))
        self.start_method = start_method
        self.dynamic_workers = bool(dynamic_workers)
        self.cpu_grow_threshold = float(cpu_grow_threshold)
        self.cpu_shrink_threshold = max(
            float(cpu_shrink_threshold), self.cpu_grow_threshold
        )
        self.cpu_check_interval = max(0.05, float(cpu_check_interval))
        self._active_worker_limit = (
            max(1, min(max_workers, int(initial_workers)))
            if self.dynamic_workers
            else max_workers
        )
        self._cpu_percent: Optional[float] = None
        self._cpu_monitor_error = ""
        self._cpu_monitor_last_at: Optional[float] = None
        self._ctx = multiprocessing.get_context(start_method)
        self._worker_counters = self._ctx.Array(
            "Q", max_workers * _WORKER_COUNTER_COUNT, lock=False
        )
        # Worker slots are provisioned lazily (see ``_ensure_worker``).  The
        # per-slot lists are pre-sized with ``None`` placeholders so a slot keeps
        # a stable index, but the process/pipe/reader-thread for a slot is only
        # created the first time the CPU-driven scheduler actually needs that
        # many concurrent workers.  This keeps the resident process/memory/fd
        # footprint proportional to ``_active_worker_limit`` instead of always
        # forking the full ``max_workers`` pool up front.
        self._provisioned = 0
        self._task_queues: list = [None] * max_workers
        self._result_readers: list = [None] * max_workers
        self._result_writers: list = [None] * max_workers
        self._condition = threading.Condition()
        self._queues: Dict[str, Deque[QueuedTask]] = {}
        self._ready_owners: Deque[str] = deque()
        self._ready_set: set[str] = set()
        self._shutdown = False
        self._result_stop = threading.Event()
        self._next_task_id = 0
        self._inflight = 0
        self._futures: Dict[int, Future] = {}
        self._queued_by_fn: Counter[str] = Counter()
        self._running_by_fn: Counter[str] = Counter()
        self._running_by_owner: Counter[str] = Counter()
        self._submitted_by_fn: Counter[str] = Counter()
        self._finished_by_fn: Counter[str] = Counter()
        self._running_tasks: Dict[int, RunningTask] = {}
        self._idle_workers: Deque[int] = deque()
        self._worker_task_started = 0
        self._worker_task_finished = 0
        self._result_received_total = 0
        self._result_completed_total = 0
        self._result_loop_error = ""
        self._result_loop_last_seen_at: Optional[float] = None
        self._result_loop_last_completed_at: Optional[float] = None
        self._result_loop_get_active = False
        self._result_reader_active = 0
        self._complete_task_active_count = 0
        self._dispatch_put_active_count = 0
        self._dispatch_put_last_ms = 0.0
        self._dispatch_put_max_ms = 0.0
        self._dispatch_put_slow_total = 0
        self._dispatch_put_slow_threshold_s = 1.0
        self._dispatch_task_active = ""
        self._dispatch_last_at: Optional[float] = None
        self._dispatch_take_none_total = 0
        self._dispatch_cancelled_before_run_total = 0
        self._dispatch_wait_total = 0
        self._workers: list = [None] * max_workers
        self._result_threads: list = [None] * max_workers

        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop,
            name="fair-eval-dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()
        self._cpu_monitor_thread: Optional[threading.Thread] = None
        if self.dynamic_workers and self.max_workers > 1:
            self._cpu_monitor_thread = threading.Thread(
                target=self._cpu_monitor_loop,
                name="fair-eval-cpu-monitor",
                daemon=True,
            )
            self._cpu_monitor_thread.start()

    def submit_eval(
        self,
        owner: str,
        fn: Callable[..., Any],
        /,
        *args: Any,
        task_priority: int = 1,
        **kwargs: Any,
    ) -> Future:
        """Submit one evaluation task for an owner/equation."""
        if not owner:
            owner = "default"
        future: Future = Future()
        fn_name = getattr(fn, "__name__", type(fn).__name__)
        if "shutdown_event" in kwargs:
            kwargs = dict(kwargs)
            kwargs["shutdown_event"] = None
        with self._condition:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            task = (
                self._next_task_id,
                future,
                fn,
                args,
                kwargs,
                fn_name,
                time.monotonic(),
                int(task_priority),
                owner,
            )
            self._next_task_id += 1
            queue = self._queues.setdefault(owner, deque())
            if task_priority <= 0:
                queue.appendleft(task)
            else:
                queue.append(task)
            self._queued_by_fn[fn_name] += 1
            self._submitted_by_fn[fn_name] += 1
            if owner not in self._ready_set:
                self._ready_owners.append(owner)
                self._ready_set.add(owner)
            self._condition.notify_all()
        return future

    def snapshot(self) -> Dict[str, Any]:
        """Return approximate executor counters for diagnostics."""
        with self._condition:
            now = time.monotonic()
            queued_ages = [
                now - task[6]
                for queue in self._queues.values()
                for task in queue
            ]
            running_ages = [
                now - started_at
                for _, _, started_at, _, _ in self._running_tasks.values()
            ]
            snap = {
                "max_workers": self.max_workers,
                "active_worker_limit": self._active_worker_limit,
                "dynamic_workers": self.dynamic_workers,
                "cpu_percent": self._cpu_percent,
                "cpu_grow_threshold": self.cpu_grow_threshold,
                "cpu_shrink_threshold": self.cpu_shrink_threshold,
                "cpu_check_interval": self.cpu_check_interval,
                "cpu_monitor_alive": (
                    self._cpu_monitor_thread.is_alive()
                    if self._cpu_monitor_thread is not None
                    else False
                ),
                "cpu_monitor_error": self._cpu_monitor_error,
                "cpu_monitor_last_age_s": (
                    now - self._cpu_monitor_last_at
                    if self._cpu_monitor_last_at is not None
                    else -1.0
                ),
                "backend": f"process:{self.start_method}",
                "result_backend": "per-worker-reader",
                "owner_max_inflight": self.owner_max_inflight,
                "owners": len(self._queues),
                "ready_owners": len(self._ready_owners),
                "queued_total": sum(len(queue) for queue in self._queues.values()),
                "running_total": sum(self._running_by_fn.values()),
                "inflight_total": self._inflight,
                "oldest_queued_age_s": max(queued_ages) if queued_ages else 0.0,
                "oldest_running_age_s": max(running_ages) if running_ages else 0.0,
                "submitted_total": sum(self._submitted_by_fn.values()),
                "finished_total": sum(self._finished_by_fn.values()),
                "worker_started": self._worker_task_started,
                "worker_finished": self._worker_task_finished,
                "idle_workers": len(self._idle_workers),
                "idle_worker_channels": len(self._idle_workers),
                "provisioned_workers": self._provisioned,
                "busy_worker_channels": self._provisioned - len(self._idle_workers),
                "input_channels": self._provisioned,
                "dispatcher_threads_alive": (
                    1 if self._dispatcher_thread.is_alive() else 0
                ),
                "dispatchers": 1,
                "dispatch_put_active_count": self._dispatch_put_active_count,
                "dispatch_put_last_ms": self._dispatch_put_last_ms,
                "dispatch_put_max_ms": self._dispatch_put_max_ms,
                "dispatch_put_slow_total": self._dispatch_put_slow_total,
                "dispatch_task_active": self._dispatch_task_active,
                "dispatch_last_age_s": (
                    now - self._dispatch_last_at
                    if self._dispatch_last_at is not None
                    else -1.0
                ),
                "dispatch_take_none_total": self._dispatch_take_none_total,
                "dispatch_cancelled_before_run_total": (
                    self._dispatch_cancelled_before_run_total
                ),
                "dispatch_wait_total": self._dispatch_wait_total,
                "result_received_total": self._result_received_total,
                "result_completed_total": self._result_completed_total,
                "result_loop_error": self._result_loop_error,
                "result_loop_get_active": self._result_reader_active > 0,
                "result_readers_active": self._result_reader_active,
                "complete_task_active": self._complete_task_active_count > 0,
                "complete_task_active_count": self._complete_task_active_count,
                "result_loop_last_seen_age_s": (
                    now - self._result_loop_last_seen_at
                    if self._result_loop_last_seen_at is not None
                    else -1.0
                ),
                "result_loop_last_completed_age_s": (
                    now - self._result_loop_last_completed_at
                    if self._result_loop_last_completed_at is not None
                    else -1.0
                ),
                "worker_processes_alive": sum(
                    1 for proc in self._workers
                    if proc is not None and proc.is_alive()
                ),
                "dispatcher_alive": self._dispatcher_thread.is_alive(),
                "result_collector_alive": all(
                    thread.is_alive()
                    for thread in self._result_threads
                    if thread is not None
                ),
                "result_readers_alive": sum(
                    1 for thread in self._result_threads
                    if thread is not None and thread.is_alive()
                ),
                "queued_by_fn": dict(self._queued_by_fn),
                "running_by_fn": dict(self._running_by_fn),
                "running_by_owner": dict(self._running_by_owner),
                "submitted_by_fn": dict(self._submitted_by_fn),
                "finished_by_fn": dict(self._finished_by_fn),
            }
        worker_exec_started = _worker_counter_sum(
            self._worker_counters, self.max_workers, _WORKER_EXEC_STARTED)
        worker_exec_finished = _worker_counter_sum(
            self._worker_counters, self.max_workers, _WORKER_EXEC_FINISHED)
        worker_put_started = _worker_counter_sum(
            self._worker_counters, self.max_workers, _WORKER_PUT_STARTED)
        worker_put_finished = _worker_counter_sum(
            self._worker_counters, self.max_workers, _WORKER_PUT_FINISHED)
        worker_put_failed = _worker_counter_sum(
            self._worker_counters, self.max_workers, _WORKER_PUT_FAILED)
        snap.update({
            "worker_exec_started": worker_exec_started,
            "worker_exec_finished": worker_exec_finished,
            "worker_put_started": worker_put_started,
            "worker_put_finished": worker_put_finished,
            "worker_put_failed": worker_put_failed,
            "worker_put_pending": max(
                0, worker_put_started - worker_put_finished - worker_put_failed),
        })
        return snap

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Signal workers to stop.

        By default queued tasks are drained before workers exit, mirroring the
        common executor behavior.  With ``cancel_futures=True``, queued but not
        yet running tasks are cancelled and discarded.
        """
        with self._condition:
            self._shutdown = True
            if cancel_futures:
                for queue in self._queues.values():
                    while queue:
                        _, future, _, _, _, fn_name, _, _, _ = queue.popleft()
                        self._queued_by_fn[fn_name] -= 1
                        if self._queued_by_fn[fn_name] <= 0:
                            self._queued_by_fn.pop(fn_name, None)
                        future.cancel()
                self._queues.clear()
                self._ready_owners.clear()
                self._ready_set.clear()
            self._condition.notify_all()

        if cancel_futures and not wait:
            for proc in self._workers:
                if proc is not None and proc.is_alive():
                    proc.terminate()
            self._result_stop.set()
            return

        if wait:
            self._dispatcher_thread.join()
            for proc in self._workers:
                if proc is not None:
                    proc.join()
            self._result_stop.set()
            for thread in self._result_threads:
                if thread is not None:
                    thread.join()
            if self._cpu_monitor_thread is not None:
                self._cpu_monitor_thread.join(timeout=1.0)

    @staticmethod
    def _read_cpu_times() -> Optional[Tuple[int, int]]:
        """Return total and idle jiffies from /proc/stat, if available."""
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                first = f.readline().strip().split()
        except Exception:
            return None
        if not first or first[0] != "cpu":
            return None
        try:
            values = [int(v) for v in first[1:]]
        except ValueError:
            return None
        if len(values) < 4:
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    @staticmethod
    def _loadavg_cpu_percent() -> Optional[float]:
        """Fallback CPU pressure estimate for non-Linux systems."""
        try:
            load1, _, _ = os.getloadavg()
            cores = os.cpu_count() or 1
            return max(0.0, min(100.0, 100.0 * float(load1) / float(cores)))
        except Exception:
            return None

    def _has_backlog_locked(self) -> bool:
        return self._has_pending_locked() or self._inflight >= self._active_worker_limit

    def _adjust_active_worker_limit(self, cpu_percent: float) -> None:
        with self._condition:
            old_limit = self._active_worker_limit
            if cpu_percent >= self.cpu_shrink_threshold:
                self._active_worker_limit = max(1, self._active_worker_limit - 1)
            elif (
                cpu_percent <= self.cpu_grow_threshold
                and self._active_worker_limit < self.max_workers
                and self._has_backlog_locked()
            ):
                self._active_worker_limit += 1
            self._cpu_percent = cpu_percent
            self._cpu_monitor_last_at = time.monotonic()
            if self._active_worker_limit != old_limit:
                self._condition.notify_all()

    def _cpu_monitor_loop(self) -> None:
        prev = self._read_cpu_times()
        ewma: Optional[float] = None
        while not self._shutdown and not self._result_stop.wait(self.cpu_check_interval):
            try:
                current = self._read_cpu_times()
                cpu_percent: Optional[float] = None
                if prev is not None and current is not None:
                    total_delta = current[0] - prev[0]
                    idle_delta = current[1] - prev[1]
                    if total_delta > 0:
                        busy = max(0, total_delta - idle_delta)
                        cpu_percent = 100.0 * float(busy) / float(total_delta)
                prev = current
                if cpu_percent is None:
                    cpu_percent = self._loadavg_cpu_percent()
                if cpu_percent is None:
                    continue
                ewma = cpu_percent if ewma is None else (0.3 * cpu_percent + 0.7 * ewma)
                self._adjust_active_worker_limit(ewma)
            except BaseException as exc:
                with self._condition:
                    self._cpu_monitor_error = repr(exc)
                    self._cpu_monitor_last_at = time.monotonic()
                time.sleep(min(1.0, self.cpu_check_interval))

    def _choose_owner_locked(self) -> str | None:
        """Choose the owner closest to clearing its current queued work.

        ``queued + running`` estimates how far an equation is from completing
        its current step.  ``owner_max_inflight`` is a soft cap: if every owner
        with queued work is at the cap, the cap is ignored rather than leaving
        global workers idle.
        """
        best_owner: str | None = None
        best_score: int | None = None
        fallback_owner: str | None = None
        fallback_score: int | None = None
        stale: list[str] = []

        for owner in list(self._ready_owners):
            queue = self._queues.get(owner)
            if not queue:
                stale.append(owner)
                continue

            score = len(queue) + self._running_by_owner.get(owner, 0)
            if fallback_score is None or score < fallback_score:
                fallback_owner = owner
                fallback_score = score

            if (
                self.owner_max_inflight > 0
                and self._running_by_owner.get(owner, 0)
                >= self.owner_max_inflight
            ):
                continue

            if best_score is None or score < best_score:
                best_owner = owner
                best_score = score

        for owner in stale:
            try:
                self._ready_owners.remove(owner)
            except ValueError:
                pass
            self._ready_set.discard(owner)
            self._queues.pop(owner, None)

        return best_owner if best_owner is not None else fallback_owner

    def _take_task_locked(self) -> Optional[QueuedTask]:
        while self._ready_owners:
            owner = self._choose_owner_locked()
            if owner is None:
                return None

            try:
                self._ready_owners.remove(owner)
            except ValueError:
                self._ready_set.discard(owner)
                continue
            self._ready_set.discard(owner)

            queue = self._queues.get(owner)
            if not queue:
                self._queues.pop(owner, None)
                continue

            task = queue.popleft()
            fn_name = task[5]
            self._queued_by_fn[fn_name] -= 1
            if self._queued_by_fn[fn_name] <= 0:
                self._queued_by_fn.pop(fn_name, None)
            if queue:
                self._ready_owners.append(owner)
                self._ready_set.add(owner)
            else:
                self._queues.pop(owner, None)
            return task
        return None

    def _has_pending_locked(self) -> bool:
        return any(self._queues.values())

    def _release_owner_locked(self, owner: str) -> None:
        self._running_by_owner[owner] -= 1
        if self._running_by_owner[owner] <= 0:
            self._running_by_owner.pop(owner, None)

    def _ensure_worker(self, worker_idx: int) -> None:
        """Lazily create and start the process/pipe/reader-thread for a slot.

        Only ever called by the dispatcher thread, exactly once per reserved
        index (the index is claimed under ``self._condition`` before this runs),
        so the per-slot list writes need no additional locking.  The heavy
        ``proc.start()`` runs outside the condition lock to avoid stalling
        ``submit_eval`` / result completion while a new worker is forked.
        """
        task_queue = self._ctx.SimpleQueue()
        result_reader, result_writer = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(
            target=_process_worker,
            args=(worker_idx, task_queue, result_writer, self._worker_counters),
            name=f"global-worker-process-{worker_idx}",
        )
        self._task_queues[worker_idx] = task_queue
        self._result_readers[worker_idx] = result_reader
        self._result_writers[worker_idx] = result_writer
        self._workers[worker_idx] = proc
        proc.start()
        result_writer.close()
        thread = threading.Thread(
            target=self._result_reader_loop,
            args=(worker_idx, result_reader),
            name=f"fair-eval-result-{worker_idx}",
            daemon=True,
        )
        self._result_threads[worker_idx] = thread
        thread.start()

    def _dispatch_loop(self) -> None:
        while True:
            task: Optional[QueuedTask] = None
            worker_idx: Optional[int] = None
            need_provision = False
            with self._condition:
                while True:
                    if self._shutdown and not self._has_pending_locked():
                        self._condition.notify_all()
                        break
                    if self._inflight < self._active_worker_limit and (
                        self._idle_workers
                        or self._provisioned < self.max_workers
                    ):
                        task = self._take_task_locked()
                        if task is not None:
                            if self._idle_workers:
                                worker_idx = self._idle_workers.popleft()
                            else:
                                worker_idx = self._provisioned
                                self._provisioned += 1
                                need_provision = True
                            break
                        if self._has_pending_locked():
                            self._dispatch_take_none_total += 1
                    self._dispatch_wait_total += 1
                    self._condition.wait(timeout=0.5)
                if task is None:
                    break

            if need_provision:
                self._ensure_worker(int(worker_idx))

            (
                task_id,
                future,
                fn,
                args,
                kwargs,
                fn_name,
                _submitted_at,
                _priority,
                owner,
            ) = task
            if not future.set_running_or_notify_cancel():
                with self._condition:
                    if worker_idx is not None:
                        self._idle_workers.appendleft(worker_idx)
                    self._dispatch_cancelled_before_run_total += 1
                    self._condition.notify_all()
                continue
            with self._condition:
                self._futures[task_id] = future
                self._running_by_fn[fn_name] += 1
                self._running_by_owner[owner] += 1
                self._running_tasks[task_id] = (
                    fn_name,
                    owner,
                    time.monotonic(),
                    _submitted_at,
                    int(worker_idx),
                )
                self._inflight += 1
                self._worker_task_started += 1
                self._dispatch_put_active_count += 1
                self._dispatch_task_active = fn_name
            put_started = time.monotonic()
            try:
                self._task_queues[int(worker_idx)].put((task_id, fn, args, kwargs))
            except BaseException as exc:
                self._complete_task(task_id, False, repr(exc))
            finally:
                elapsed_ms = (time.monotonic() - put_started) * 1000.0
                with self._condition:
                    self._dispatch_put_active_count = max(
                        0, self._dispatch_put_active_count - 1
                    )
                    self._dispatch_put_last_ms = elapsed_ms
                    self._dispatch_put_max_ms = max(
                        self._dispatch_put_max_ms, elapsed_ms
                    )
                    if elapsed_ms >= self._dispatch_put_slow_threshold_s * 1000.0:
                        self._dispatch_put_slow_total += 1
                    self._dispatch_task_active = ""
                    self._dispatch_last_at = time.monotonic()

        for task_queue in self._task_queues:
            if task_queue is not None:
                task_queue.put(None)

    def _result_reader_loop(self, worker_idx: int, conn: Any) -> None:
        try:
            while True:
                with self._condition:
                    self._result_reader_active += 1
                    self._result_loop_get_active = True
                try:
                    item = conn.recv()
                except EOFError:
                    return
                finally:
                    with self._condition:
                        self._result_reader_active -= 1
                        self._result_loop_get_active = self._result_reader_active > 0
                with self._condition:
                    self._result_received_total += 1
                    self._result_loop_last_seen_at = time.monotonic()
                task_id, ok, payload = item
                self._complete_task(task_id, ok, payload)
                with self._condition:
                    self._result_completed_total += 1
                    self._result_loop_last_completed_at = time.monotonic()
        except BaseException as exc:
            with self._condition:
                self._result_loop_error = f"reader[{worker_idx}]: {exc!r}"
                self._result_loop_get_active = self._result_reader_active > 0
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _complete_task(self, task_id: int, ok: bool, payload: Any) -> None:
        future: Optional[Future]
        with self._condition:
            self._complete_task_active_count += 1
        try:
            with self._condition:
                meta = self._running_tasks.pop(task_id, None)
                future = self._futures.pop(task_id, None)
                if meta is not None:
                    fn_name, owner, _, _, worker_idx = meta
                    self._running_by_fn[fn_name] -= 1
                    if self._running_by_fn[fn_name] <= 0:
                        self._running_by_fn.pop(fn_name, None)
                    self._release_owner_locked(owner)
                    self._idle_workers.append(worker_idx)
                    self._finished_by_fn[fn_name] += 1
                    self._worker_task_finished += 1
                    self._inflight -= 1
                self._condition.notify_all()

            if future is None:
                return
            if ok:
                future.set_result(payload)
            else:
                future.set_exception(RuntimeError(f"FairEvalExecutor worker failed: {payload}"))
        finally:
            with self._condition:
                self._complete_task_active_count -= 1
