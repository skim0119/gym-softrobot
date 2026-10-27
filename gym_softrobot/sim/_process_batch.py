import multiprocessing
import os
import signal
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

import numpy as np
from numpy.typing import NDArray

from gym_softrobot.sim.batch import (
    BatchFrame,
    BatchStep,
    ProcessPool,
    RodBatch,
    TaskSpec,
    require_actions,
    require_generators,
)
from gym_softrobot.sim.elastica import ElasticaTask

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
)
_JOIN_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class Layout:
    """Contiguous env blocks per worker, and the worker that owns each env."""

    spans: tuple[range, ...]
    owner: tuple[int, ...]


def _layout(num_envs: int, workers: int) -> Layout:
    base, extra = divmod(num_envs, workers)
    spans = []
    start = 0
    for worker in range(workers):
        stop = start + base + (worker < extra)
        spans.append(range(start, stop))
        start = stop
    owner = tuple(worker for worker, span in enumerate(spans) for _ in span)
    return Layout(spans=tuple(spans), owner=owner)


@dataclass(frozen=True, slots=True)
class _Reset:
    slots: tuple[int, ...]
    rngs: tuple[np.random.Generator, ...]


@dataclass(frozen=True, slots=True)
class _Step:
    actions: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _Frame:
    pass


@dataclass(frozen=True, slots=True)
class _Close:
    pass


@dataclass(frozen=True, slots=True)
class _Reply:
    value: object


@dataclass(frozen=True, slots=True)
class _Raised:
    error: BaseException


def _lost_step(spec: TaskSpec) -> BatchStep:
    return BatchStep(
        obs=np.zeros((1, spec.observation_space.shape[0]), dtype=np.float32),
        reward=np.zeros(1, dtype=np.float32),
        failed=np.ones(1, dtype=np.bool_),
    )


def _lost_frame(spec: TaskSpec, count: int) -> BatchFrame:
    return BatchFrame(
        centerlines=tuple(
            np.zeros((count, 3, n), dtype=np.float64) for n in spec.rod_nodes
        ),
        radii=tuple(np.zeros((count, n - 1), dtype=np.float64) for n in spec.rod_nodes),
    )


def _stack_frames(frames: Sequence[BatchFrame]) -> BatchFrame:
    return BatchFrame(
        centerlines=tuple(
            np.concatenate(rod) for rod in zip(*(f.centerlines for f in frames))
        ),
        radii=tuple(np.concatenate(rod) for rod in zip(*(f.radii for f in frames))),
    )


def _handle(
    task: ElasticaTask, slots: list[RodBatch | None], command: object
) -> object:
    spec = task.spec
    match command:
        case _Reset(slots=ids, rngs=rngs):
            obs = []
            for slot, rng in zip(ids, rngs, strict=True):
                batch = task.make_batch(1)
                obs.append(batch.reset(np.zeros(1, dtype=np.intp), [rng])[0])
                slots[slot] = batch
            return np.stack(obs), tuple(rng.bit_generator.state for rng in rngs)
        case _Step(actions=actions):
            steps = [
                _lost_step(spec)
                if batch is None
                else batch.step(actions[row : row + 1])
                for row, batch in enumerate(slots)
            ]
            return BatchStep(
                obs=np.concatenate([step.obs for step in steps]),
                reward=np.concatenate([step.reward for step in steps]),
                failed=np.concatenate([step.failed for step in steps]),
            )
        case _Frame():
            return _stack_frames(
                [
                    _lost_frame(spec, 1) if batch is None else batch.frame()
                    for batch in slots
                ]
            )
    raise TypeError(f"unknown command {command!r}")


def _serve(conn: Connection, task: ElasticaTask, slot_count: int) -> None:
    # Ctrl-C belongs to the parent, which closes the pipes to stop the workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    slots: list[RodBatch | None] = [None] * slot_count
    while True:
        try:
            command = conn.recv()
        except EOFError:
            return
        if isinstance(command, _Close):
            return
        try:
            reply: _Reply | _Raised = _Reply(_handle(task, slots, command))
        except Exception as error:
            reply = _Raised(error)
        try:
            conn.send(reply)
        except Exception as error:
            # An unpicklable reply would otherwise kill the worker and read as a lost env.
            unsent = repr(reply.error) if isinstance(reply, _Raised) else "its result"
            conn.send(
                _Raised(RuntimeError(f"rod worker could not send {unsent}: {error!r}"))
            )


@dataclass(slots=True)
class _Worker:
    process: BaseProcess
    conn: Connection


class ProcessBatch:
    """An ElasticaBatch split across spawned processes, one slot per env."""

    def __init__(self, task: ElasticaTask, num_envs: int, pool: ProcessPool) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self._task = task
        self._num_envs = num_envs
        self._layout = _layout(
            num_envs, min(pool.workers or os.cpu_count() or 1, num_envs)
        )
        self._context = multiprocessing.get_context("spawn")
        self._was_reset = np.zeros(num_envs, dtype=np.bool_)
        self._workers = [self._start(len(span)) for span in self._layout.spans]
        self._closed = False

    @property
    def spec(self) -> TaskSpec:
        return self._task.spec

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(
        self, env_ids: NDArray[np.intp], rngs: Sequence[np.random.Generator]
    ) -> NDArray[np.float32]:
        if self._closed:
            raise RuntimeError("batch is closed")
        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        require_generators(ids, rngs)
        if ((ids < -self._num_envs) | (ids >= self._num_envs)).any():
            raise IndexError(f"env ids out of range for {self._num_envs} envs: {ids}")
        ids = ids % self._num_envs
        rows: list[list[int]] = [[] for _ in self._layout.spans]
        for row, index in enumerate(ids):
            rows[self._layout.owner[index]].append(row)
        commands = {
            worker: _Reset(
                slots=tuple(
                    int(ids[row]) - self._layout.spans[worker].start for row in owned
                ),
                rngs=tuple(rngs[row] for row in owned),
            )
            for worker, owned in enumerate(rows)
            if owned
        }
        replies = self._exchange(commands)
        retry = {
            worker: commands[worker]
            for worker, reply in replies.items()
            if reply is None
        }
        if retry:
            replies |= self._exchange(retry)
        obs = np.empty(
            (ids.size, int(self.spec.observation_space.shape[0])), dtype=np.float32
        )
        for worker, reply in replies.items():
            if not isinstance(reply, _Reply):
                continue
            worker_obs, states = reply.value
            for row, local, state in zip(rows[worker], worker_obs, states, strict=True):
                obs[row] = local
                rngs[row].bit_generator.state = state
                self._was_reset[ids[row]] = True
        if any(reply is None for reply in replies.values()):
            raise RuntimeError("a rod worker died twice while resetting")
        _raise_first(replies.values())
        return obs

    def step(self, actions: NDArray[np.float32]) -> BatchStep:
        self._require_ready()
        actions = require_actions(actions, self._num_envs, self.spec)
        replies = self._exchange(
            {
                worker: _Step(np.ascontiguousarray(actions[span.start : span.stop]))
                for worker, span in enumerate(self._layout.spans)
            }
        )
        _raise_first(replies.values())
        obs = np.zeros(
            (self._num_envs, int(self.spec.observation_space.shape[0])),
            dtype=np.float32,
        )
        reward = np.zeros(self._num_envs, dtype=np.float32)
        failed = np.ones(self._num_envs, dtype=np.bool_)
        for worker, reply in replies.items():
            if reply is None:
                continue
            span = self._layout.spans[worker]
            obs[span.start : span.stop] = reply.value.obs
            reward[span.start : span.stop] = reply.value.reward
            failed[span.start : span.stop] = reply.value.failed
        return BatchStep(obs=obs, reward=reward, failed=failed)

    def frame(self) -> BatchFrame:
        self._require_ready()
        replies = self._exchange(
            {worker: _Frame() for worker in range(len(self._workers))}
        )
        _raise_first(replies.values())
        return _stack_frames(
            [
                _lost_frame(self.spec, len(self._layout.spans[worker]))
                if reply is None
                else reply.value
                for worker, reply in sorted(replies.items())
            ]
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._was_reset[:] = False
        for worker in self._workers:
            try:
                worker.conn.send(_Close())
            except OSError:
                pass
        for worker in self._workers:
            worker.process.join(_JOIN_TIMEOUT_S)
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join()
            worker.conn.close()

    def _require_ready(self) -> None:
        missing = np.flatnonzero(~self._was_reset).tolist()
        if missing:
            raise RuntimeError(f"worlds were never reset: {missing}")

    def _start(self, slot_count: int) -> _Worker:
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=_serve, args=(child, self._task, slot_count), daemon=True
        )
        # Spawned children read these before importing NumPy; setting them later has no effect.
        saved = {name: os.environ.get(name) for name in _THREAD_VARS}
        os.environ.update(dict.fromkeys(_THREAD_VARS, "1"))
        try:
            process.start()
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        child.close()
        return _Worker(process=process, conn=parent)

    def _exchange(
        self, commands: dict[int, object]
    ) -> dict[int, _Reply | _Raised | None]:
        """Send every command, then drain every reply. ``None`` marks a worker that died."""
        sent = {}
        for worker, command in commands.items():
            try:
                self._workers[worker].conn.send(command)
                sent[worker] = True
            except OSError:
                sent[worker] = False
        replies: dict[int, _Reply | _Raised | None] = {}
        for worker in commands:
            try:
                replies[worker] = (
                    self._workers[worker].conn.recv() if sent[worker] else None
                )
            except (EOFError, OSError):
                replies[worker] = None
            if replies[worker] is None:
                self._replace(worker)
        return replies

    def _replace(self, worker: int) -> None:
        dead = self._workers[worker]
        dead.conn.close()
        dead.process.join(_JOIN_TIMEOUT_S)
        if dead.process.is_alive():
            dead.process.kill()
            dead.process.join()
        self._workers[worker] = self._start(len(self._layout.spans[worker]))


def _raise_first(replies: Iterable[_Reply | _Raised | None]) -> None:
    for reply in replies:
        if isinstance(reply, _Raised):
            raise reply.error
