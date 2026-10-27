import os
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from gym_softrobot.sim import ProcessPool
from gym_softrobot.sim.batch import BatchFrame, BatchStep, RodBatch
from gym_softrobot.sim.elastica import ElasticaBatch
from test_rod_batch import (
    FakeTask,
    _first_uniform,
    _ids,
    _NanOnPositive,
    _ShortRod,
    _World,
)


class _ExitOnPositive(FakeTask):
    def apply_action(self, world: _World, action: NDArray[np.float32]) -> None:
        if float(action[0]) > 0.0:
            os._exit(1)
        super().apply_action(world, action)


class _WrongObservation(FakeTask):
    def observe(self, world: _World) -> NDArray[np.float32]:
        return np.zeros(3, dtype=np.float32)


@pytest.fixture
def batches() -> Iterator[list[RodBatch]]:
    opened: list[RodBatch] = []
    yield opened
    for batch in opened:
        batch.close()


def _open(opened: list[RodBatch], batch: RodBatch) -> RodBatch:
    opened.append(batch)
    return batch


def _rngs(*seeds: int) -> list[np.random.Generator]:
    return [np.random.default_rng(seed) for seed in seeds]


def _actions(*values: float) -> np.ndarray:
    return np.array([[value] for value in values], dtype=np.float32)


def _assert_same_step(actual: BatchStep, expected: BatchStep) -> None:
    np.testing.assert_array_equal(actual.obs, expected.obs)
    np.testing.assert_array_equal(actual.reward, expected.reward)
    np.testing.assert_array_equal(actual.failed, expected.failed)


def _assert_same_frame(actual: BatchFrame, expected: BatchFrame) -> None:
    for got, want in zip(actual.centerlines, expected.centerlines, strict=True):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(actual.radii, expected.radii, strict=True):
        np.testing.assert_array_equal(got, want)


def test_make_batch_is_serial_unless_given_a_pool(batches) -> None:
    assert isinstance(_open(batches, FakeTask().make_batch(2)), ElasticaBatch)
    process = _open(batches, FakeTask().make_batch(2, ProcessPool(workers=2)))
    assert not isinstance(process, ElasticaBatch)
    assert process.num_envs == 2


@pytest.mark.parametrize("workers", [True, 0, -1, 1.5])
def test_process_pool_rejects_bad_worker_counts(workers) -> None:
    with pytest.raises(ValueError):
        ProcessPool(workers=workers)


def test_reset_step_and_frame_match_serial(batches) -> None:
    serial = _open(batches, FakeTask().make_batch(3))
    process = _open(batches, FakeTask().make_batch(3, ProcessPool(workers=2)))
    serial_rngs, process_rngs = _rngs(0, 1, 2), _rngs(0, 1, 2)
    np.testing.assert_array_equal(
        process.reset(_ids(0, 1, 2), process_rngs),
        serial.reset(_ids(0, 1, 2), serial_rngs),
    )
    assert [rng.random() for rng in process_rngs] == [
        rng.random() for rng in serial_rngs
    ]
    for actions in (_actions(-1.0, 0.5, 1.0), _actions(0.0, 0.0, 0.0)):
        _assert_same_step(process.step(actions), serial.step(actions))
    _assert_same_frame(process.frame(), serial.frame())
    np.testing.assert_array_equal(
        process.reset(_ids(2, 0), _rngs(7, 8)), serial.reset(_ids(2, 0), _rngs(7, 8))
    )
    _assert_same_step(process.step(_actions(0, 0, 0)), serial.step(_actions(0, 0, 0)))
    _assert_same_frame(process.frame(), serial.frame())


def test_partial_reset_changes_only_reset_rows(batches) -> None:
    process = _open(batches, FakeTask().make_batch(3, ProcessPool(workers=2)))
    process.reset(_ids(0, 1, 2), _rngs(0, 1, 2))
    before = process.frame()
    obs = process.reset(_ids(1), _rngs(5))
    after = process.frame()
    np.testing.assert_allclose(obs, [[_first_uniform(5), 0.0]])
    np.testing.assert_allclose(after.centerlines[0][1], _first_uniform(5))
    for row in (0, 2):
        np.testing.assert_array_equal(
            after.centerlines[0][row], before.centerlines[0][row]
        )


def test_nonfinite_velocity_stays_failed_until_reset(batches) -> None:
    serial = _open(batches, _NanOnPositive().make_batch(2))
    process = _open(batches, _NanOnPositive().make_batch(2, ProcessPool(workers=2)))
    for batch in (serial, process):
        batch.reset(_ids(0, 1), _rngs(0, 1))
    first = process.step(_actions(-1.0, 1.0))
    _assert_same_step(first, serial.step(_actions(-1.0, 1.0)))
    assert first.failed.tolist() == [False, True]
    np.testing.assert_array_equal(first.obs[1], [0.0, 0.0])
    second = process.step(_actions(-1.0, -1.0))
    _assert_same_step(second, serial.step(_actions(-1.0, -1.0)))
    assert second.failed.tolist() == [False, True]
    for batch in (serial, process):
        batch.reset(_ids(1), _rngs(3))
    third = process.step(_actions(-1.0, -1.0))
    _assert_same_step(third, serial.step(_actions(-1.0, -1.0)))
    assert third.failed.tolist() == [False, False]


def test_dead_worker_fails_only_its_rows_until_reset(batches) -> None:
    serial = _open(batches, FakeTask().make_batch(2))
    process = _open(batches, _ExitOnPositive().make_batch(2, ProcessPool(workers=2)))
    for batch in (serial, process):
        batch.reset(_ids(0, 1), _rngs(0, 1))

    lost = process.step(_actions(-1.0, 1.0))
    expected = serial.step(_actions(-1.0, 1.0))
    assert lost.failed.tolist() == [False, True]
    np.testing.assert_array_equal(lost.obs[0], expected.obs[0])
    assert lost.reward[0] == expected.reward[0]
    np.testing.assert_array_equal(lost.obs[1], [0.0, 0.0])
    assert lost.reward[1] == 0.0
    np.testing.assert_array_equal(process.frame().centerlines[0][1], np.zeros((3, 3)))

    still_lost = process.step(_actions(-1.0, -1.0))
    expected = serial.step(_actions(-1.0, -1.0))
    assert still_lost.failed.tolist() == [False, True]
    np.testing.assert_array_equal(still_lost.obs[0], expected.obs[0])

    np.testing.assert_array_equal(
        process.reset(_ids(1), _rngs(4)), serial.reset(_ids(1), _rngs(4))
    )
    _assert_same_step(
        process.step(_actions(-1.0, -1.0)), serial.step(_actions(-1.0, -1.0))
    )
    _assert_same_frame(process.frame(), serial.frame())


def test_bad_actions_and_task_errors_surface_as_value_error(batches) -> None:
    process = _open(batches, FakeTask().make_batch(2, ProcessPool(workers=2)))
    with pytest.raises(RuntimeError):
        process.step(_actions(0.0, 0.0))
    process.reset(_ids(0, 1), _rngs(0, 1))
    with pytest.raises(ValueError, match=r"actions must have shape \(2, 1\)"):
        process.step(_actions(0.0, 0.0, 0.0))
    for task in (_ShortRod(), _WrongObservation()):
        broken = _open(batches, task.make_batch(2, ProcessPool(workers=2)))
        with pytest.raises(ValueError):
            broken.reset(_ids(0, 1), _rngs(0, 1))


def test_close_is_idempotent_and_ends_the_batch() -> None:
    process = FakeTask().make_batch(2, ProcessPool(workers=2))
    process.reset(_ids(0, 1), _rngs(0, 1))
    process.close()
    process.close()
    with pytest.raises(RuntimeError):
        process.step(_actions(0.0, 0.0))


def test_snake_control_step_matches_serial(batches) -> None:
    from gym_softrobot.envs.snake.locomotion_task import SnakeLocomotionTask

    task = SnakeLocomotionTask()
    serial = _open(batches, task.make_batch(2))
    process = _open(batches, task.make_batch(2, ProcessPool(workers=2)))
    space = task.spec.action_space
    actions = np.stack([space.high, space.low]).astype(np.float32)
    np.testing.assert_array_equal(
        process.reset(_ids(0, 1), _rngs(0, 1)), serial.reset(_ids(0, 1), _rngs(0, 1))
    )
    expected = serial.step(actions)
    assert not expected.failed.any()
    _assert_same_step(process.step(actions), expected)
