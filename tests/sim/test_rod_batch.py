import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from numpy.typing import NDArray

from gym_softrobot.sim import (
    BatchFrame,
    ElasticaTask,
    ElasticaWorld,
    RodTaskEnv,
    TaskSpec,
)

CONTROL_DT = 0.2
NODES = 3


def _spec(
    *,
    action_space: spaces.Box | None = None,
    observation_space: spaces.Box | None = None,
    control_dt: float = CONTROL_DT,
    horizon: int = 4,
    rod_nodes: tuple[int, ...] = (NODES,),
) -> TaskSpec:
    if action_space is None:
        action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    if observation_space is None:
        observation_space = spaces.Box(-50.0, 50.0, shape=(2,), dtype=np.float32)
    return TaskSpec(
        action_space=action_space,
        observation_space=observation_space,
        control_dt=control_dt,
        horizon=horizon,
        rod_nodes=rod_nodes,
    )


def _first_uniform(seed: int) -> float:
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    return float(generator.random())


class _Stepper:
    def step(self, simulator, time: float, dt: float) -> float:
        for rod in simulator.rods:
            rod.position_collection += dt
        return time + dt


class _World(ElasticaWorld):
    steps: int = 0


def _make_rod(n_nodes: int, token: float):
    from types import SimpleNamespace

    return SimpleNamespace(
        position_collection=np.full((3, n_nodes), token, dtype=np.float64),
        velocity_collection=np.zeros((3, n_nodes), dtype=np.float64),
        director_collection=np.repeat(np.eye(3)[:, :, None], n_nodes - 1, axis=2),
        radius=np.full(n_nodes - 1, 0.01, dtype=np.float64),
    )


class FakeTask(ElasticaTask[_World]):
    spec: TaskSpec = _spec()
    fail_at: int | None = None

    def __init__(
        self, *, horizon: int | None = None, fail_at: int | None = None
    ) -> None:
        if horizon is not None:
            self.spec = _spec(horizon=horizon)
        if fail_at is not None:
            self.fail_at = fail_at

    def build(self, rng: np.random.Generator) -> _World:
        from types import SimpleNamespace

        token = float(rng.random())
        rods = tuple(_make_rod(n_nodes, token) for n_nodes in self.spec.rod_nodes)
        return _World(
            simulator=SimpleNamespace(rods=rods),
            timestepper=_Stepper(),
            dt=self.spec.control_dt,
            rods=rods,
        )

    def apply_action(self, world: _World, action: NDArray[np.float32]) -> None:
        del action
        world.steps += 1

    def observe(self, world: _World) -> NDArray[np.float32]:
        position = world.rods[0].position_collection
        return np.array([position[0, 0], world.steps], dtype=np.float32)

    def reward(self, world: _World) -> float:
        return float(world.rods[0].position_collection[0, 0])

    def failed(self, world: _World) -> bool:
        return self.fail_at is not None and world.steps >= self.fail_at


class _NanOnPositive(FakeTask):
    def apply_action(self, world: _World, action: NDArray[np.float32]) -> None:
        super().apply_action(world, action)
        if float(action[0]) > 0.0:
            world.rods[0].velocity_collection[0, 0] = np.nan


class _ShortRod(FakeTask):
    def build(self, rng: np.random.Generator) -> _World:
        world = super().build(rng)
        world.rods[0].position_collection = np.zeros((3, 2), dtype=np.float64)
        world.rods[0].radius = np.zeros(1, dtype=np.float64)
        return world


class _FractionalDt(FakeTask):
    def build(self, rng: np.random.Generator) -> _World:
        world = super().build(rng)
        world.dt = self.spec.control_dt / 2.5
        return world


class _ReuseBuffer(FakeTask):
    def __init__(self) -> None:
        self._buffer: np.ndarray | None = None

    def build(self, rng: np.random.Generator) -> _World:
        world = super().build(rng)
        rod = world.rods[0]
        if self._buffer is None:
            self._buffer = rod.position_collection.copy()
        else:
            self._buffer[...] = rod.position_collection
        rod.position_collection = self._buffer
        return world


class _Boom(FakeTask):
    def __init__(self) -> None:
        self._calls = 0

    def build(self, rng: np.random.Generator) -> _World:
        self._calls += 1
        if self._calls > 1:
            raise RuntimeError("build failed")
        return super().build(rng)


def _action() -> np.ndarray:
    return np.zeros(1, dtype=np.float32)


def _ids(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.intp)


def test_check_env_accepts_rod_task_env() -> None:
    env = RodTaskEnv(FakeTask())
    check_env(env)
    env.close()


def test_failure_before_horizon_terminates() -> None:
    env = RodTaskEnv(FakeTask(horizon=4, fail_at=2))
    env.reset(seed=3)
    token = _first_uniform(3)
    _, reward, terminated, truncated, info = env.step(_action())
    assert info == {}
    assert type(reward) is float
    assert reward == pytest.approx(token + CONTROL_DT)
    assert terminated is False
    assert truncated is False
    obs, reward, terminated, truncated, info = env.step(_action())
    assert info == {}
    assert terminated is True
    assert truncated is False
    assert type(terminated) is bool
    assert type(truncated) is bool
    assert reward == pytest.approx(token + 2 * CONTROL_DT)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs[0], token + 2 * CONTROL_DT)
    assert obs[1] == 2
    env.close()


def test_horizon_without_failure_truncates() -> None:
    env = RodTaskEnv(FakeTask(horizon=3))
    env.reset(seed=4)
    token = _first_uniform(4)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(_action())
        assert terminated is False
        assert truncated is False
    obs, reward, terminated, truncated, info = env.step(_action())
    assert info == {}
    assert terminated is False
    assert truncated is True
    assert type(terminated) is bool
    assert type(truncated) is bool
    assert reward == pytest.approx(token + 3 * CONTROL_DT)
    np.testing.assert_allclose(obs[0], token + 3 * CONTROL_DT)
    assert obs[1] == 3
    env.close()


def test_failure_on_horizon_sets_both_flags() -> None:
    env = RodTaskEnv(FakeTask(horizon=3, fail_at=3))
    env.reset(seed=5)
    token = _first_uniform(5)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(_action())
        assert terminated is False
        assert truncated is False
    obs, reward, terminated, truncated, info = env.step(_action())
    assert info == {}
    assert terminated is True
    assert truncated is True
    assert reward == pytest.approx(token + 3 * CONTROL_DT)
    np.testing.assert_allclose(obs[0], token + 3 * CONTROL_DT)
    env.close()


def test_nonfinite_velocity_fails_only_that_world_with_finite_outputs() -> None:
    batch = _NanOnPositive().make_batch(2)
    tokens = (_first_uniform(0), _first_uniform(1))
    batch.reset(_ids(0, 1), [np.random.default_rng(0), np.random.default_rng(1)])
    result = batch.step(np.array([[-1.0], [1.0]], dtype=np.float32))
    assert result.failed.tolist() == [False, True]
    assert np.isfinite(result.obs[0]).all()
    assert np.isfinite(result.reward[0])
    assert result.reward[0] == pytest.approx(np.float32(tokens[0] + CONTROL_DT))
    assert np.isfinite(result.obs[1]).all()
    np.testing.assert_array_equal(result.obs[1], np.zeros(2, dtype=np.float32))
    assert result.reward[1] == 0.0
    batch.close()


def test_reset_one_env_leaves_the_other_frame() -> None:
    batch = FakeTask().make_batch(2)
    batch.reset(_ids(0, 1), [np.random.default_rng(0), np.random.default_rng(1)])
    before = batch.frame()
    batch.reset(_ids(1), [np.random.default_rng(2)])
    after = batch.frame()
    np.testing.assert_array_equal(after.centerlines[0][0], before.centerlines[0][0])
    np.testing.assert_array_equal(after.radii[0][0], before.radii[0][0])
    np.testing.assert_allclose(after.centerlines[0][1], _first_uniform(2))
    assert not np.array_equal(after.centerlines[0][1], before.centerlines[0][1])
    batch.close()


def test_frame_after_reset_is_new_and_previous_frame_is_a_copy() -> None:
    batch = _ReuseBuffer().make_batch(1)
    batch.reset(_ids(0), [np.random.default_rng(1)])
    before = batch.frame()
    first = _first_uniform(1)
    np.testing.assert_allclose(before.centerlines[0], first)
    batch.reset(_ids(0), [np.random.default_rng(2)])
    after = batch.frame()
    second = _first_uniform(2)
    np.testing.assert_allclose(after.centerlines[0], second)
    np.testing.assert_allclose(before.centerlines[0], first)
    batch.close()


def test_reset_rejects_wrong_layout_or_noninteger_dt() -> None:
    short = _ShortRod().make_batch(1)
    with pytest.raises(ValueError):
        short.reset(_ids(0), [np.random.default_rng(0)])
    fractional = _FractionalDt().make_batch(1)
    with pytest.raises(ValueError):
        fractional.reset(_ids(0), [np.random.default_rng(0)])


def test_task_spec_rejects_non_float32_box() -> None:
    with pytest.raises(ValueError):
        _spec(
            observation_space=spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
        )
    with pytest.raises(ValueError):
        _spec(action_space=spaces.Box(-1.0, 1.0, shape=(1, 1), dtype=np.float32))
    with pytest.raises(ValueError):
        _spec(control_dt=0.0)
    with pytest.raises(ValueError):
        _spec(horizon=0)
    with pytest.raises(ValueError):
        _spec(rod_nodes=(1,))
    spec = _spec()
    assert spec.episode_length_s == spec.horizon * spec.control_dt


def test_batch_frame_rejects_vector() -> None:
    with pytest.raises(ValueError):
        BatchFrame(
            centerlines=(np.zeros(5, dtype=np.float64),),
            radii=(np.zeros(4, dtype=np.float64),),
        )


def test_failed_build_keeps_the_previous_frame() -> None:
    batch = _Boom().make_batch(1)
    batch.reset(_ids(0), [np.random.default_rng(0)])
    before = batch.frame()
    np.testing.assert_allclose(before.centerlines[0], _first_uniform(0))
    with pytest.raises(RuntimeError):
        batch.reset(_ids(0), [np.random.default_rng(1)])
    after = batch.frame()
    np.testing.assert_array_equal(after.centerlines[0], before.centerlines[0])
    np.testing.assert_array_equal(after.radii[0], before.radii[0])
    batch.close()


def test_reset_ignores_global_numpy_generator() -> None:
    env = RodTaskEnv(FakeTask())
    first, info = env.reset(seed=11)
    assert info == {}
    np.random.random()
    second, _ = env.reset(seed=11)
    expected = np.array([_first_uniform(11), 0.0], dtype=np.float32)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, expected)
    env.close()


def test_step_before_reset_raises() -> None:
    batch = FakeTask().make_batch(1)
    with pytest.raises(RuntimeError):
        batch.step(np.zeros((1, 1), dtype=np.float32))


def test_step_rejects_wrong_action_batch_shape() -> None:
    batch = FakeTask().make_batch(2)
    batch.reset(_ids(0, 1), [np.random.default_rng(0), np.random.default_rng(1)])
    with pytest.raises(ValueError, match=r"actions must have shape \(2, 1\)"):
        batch.step(np.zeros((3, 1), dtype=np.float32))
    with pytest.raises(ValueError, match=r"actions must have shape \(2, 1\)"):
        batch.step(np.zeros((1, 1), dtype=np.float32))
    batch.close()


def test_render_rgb_array_is_uint8_and_none_mode_returns_none() -> None:
    blank = RodTaskEnv(FakeTask())
    blank.reset(seed=0)
    assert blank.render() is None
    blank.close()

    env = RodTaskEnv(FakeTask(), render_mode="rgb_array")
    env.reset(seed=0)
    image = env.render()
    assert image is not None
    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.min() != image.max()
    env.close()


def test_render_mode_must_be_rgb_array() -> None:
    with pytest.raises(ValueError):
        RodTaskEnv(FakeTask(), render_mode="human")
