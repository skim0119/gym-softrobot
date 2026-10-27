from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import MISSING
from typing import Any

import numpy as np
import torch
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from pxr import Gf, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils

from gym_softrobot.sim.batch import BatchFrame, BatchStep, ProcessPool, SoftRobotTask

__all__ = ["SoftRobotEnv", "SoftRobotEnvCfg"]


def _load_task(entry: str) -> SoftRobotTask:
    module_name, separator, class_name = entry.partition(":")
    if separator != ":" or not module_name or not class_name:
        raise ValueError(f"task_entry must be 'module:Class', got {entry!r}")
    module = importlib.import_module(module_name)
    task_cls = getattr(module, class_name)
    return task_cls()


def _as_indices(env_ids: Sequence[int] | torch.Tensor) -> np.ndarray:
    if isinstance(env_ids, torch.Tensor):
        array = env_ids.detach().cpu().numpy()
    else:
        array = np.asarray(env_ids)
    return np.asarray(array, dtype=np.intp).reshape(-1)


def _episode_generator(
    seed: int | None, env_index: int, episode: int
) -> np.random.Generator:
    sequence = np.random.SeedSequence(seed, spawn_key=(env_index, episode))
    return np.random.Generator(np.random.PCG64(sequence))


def _vertex_widths_from_element_radii(radii: np.ndarray) -> list[float]:
    radii = np.asarray(radii, dtype=np.float64)
    nodes = np.empty(radii.shape[0] + 1, dtype=np.float64)
    nodes[0] = radii[0]
    nodes[-1] = radii[-1]
    if nodes.shape[0] > 2:
        nodes[1:-1] = 0.5 * (radii[:-1] + radii[1:])
    return (2.0 * nodes).tolist()


def _curve_with_count(stage: Usd.Stage, path: str, n_points: int) -> UsdGeom.BasisCurves:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        curves = UsdGeom.BasisCurves(prim)
        counts = curves.GetCurveVertexCountsAttr().Get()
        if counts is not None and list(counts) == [n_points]:
            return curves
        # A BasisCurves vertex count is fixed, so a new node count needs a new prim.
        stage.RemovePrim(Sdf.Path(path))
    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr("linear")
    curves.CreateCurveVertexCountsAttr([n_points])
    curves.CreatePointsAttr()
    curves.CreateWidthsAttr()
    curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    return curves


def _write_curves(
    stage: Usd.Stage,
    prim_paths: tuple[tuple[str, ...], ...],
    frame: BatchFrame,
    origins: np.ndarray,
) -> None:
    prepared: list[tuple[UsdGeom.BasisCurves, list[Gf.Vec3f], list[float]]] = []
    for env_index, rod_paths in enumerate(prim_paths):
        origin = np.asarray(origins[env_index], dtype=np.float64).reshape(3, 1)
        for rod_index, path in enumerate(rod_paths):
            center = np.asarray(frame.centerlines[rod_index][env_index], dtype=np.float64)
            points = center + origin
            curves = _curve_with_count(stage, path, int(points.shape[1]))
            vectors = [
                Gf.Vec3f(float(points[0, node]), float(points[1, node]), float(points[2, node]))
                for node in range(points.shape[1])
            ]
            widths = _vertex_widths_from_element_radii(
                frame.radii[rod_index][env_index]
            )
            prepared.append((curves, vectors, widths))
    with Sdf.ChangeBlock():
        for curves, vectors, widths in prepared:
            curves.GetPointsAttr().Set(vectors)
            curves.GetWidthsAttr().Set(widths)


@configclass
class SoftRobotEnvCfg(DirectRLEnvCfg):
    """Subclasses set ``task_entry`` to ``module:Class``.

    Spaces, ``sim.dt``, decimation, and ``episode_length_s`` are copied from
    that task's ``TaskSpec`` when the environment is constructed.
    """

    task_entry: str = MISSING
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=1.0)
    # PhysX only clocks an empty stage. The rod control interval is about 0.2 s,
    # and Isaac Lab warns on that step unless these solver flags are set.
    sim: SimulationCfg = SimulationCfg(
        render_interval=1,
        physx=PhysxCfg(
            enable_stabilization=True,
            enable_external_forces_every_iteration=True,
        ),
    )
    decimation: int = 1
    rod_processes: bool = False
    rod_workers: int | None = None


class SoftRobotEnv(DirectRLEnv):
    """Isaac Lab front end over a RodBatch.

    PhysX advances an empty stage as a clock. The batch is the physics.
    USD BasisCurves show the rods.
    """

    cfg: SoftRobotEnvCfg
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self, cfg: SoftRobotEnvCfg, render_mode: str | None = None, **kwargs: Any
    ) -> None:
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        task = _load_task(cfg.task_entry)
        spec = task.spec
        cfg.action_space = spec.action_space
        cfg.observation_space = spec.observation_space
        cfg.sim.dt = spec.control_dt
        cfg.decimation = 1
        cfg.episode_length_s = spec.episode_length_s
        pool = ProcessPool(workers=cfg.rod_workers) if cfg.rod_processes else None
        self._batch = task.make_batch(cfg.scene.num_envs, pool)
        self._seed = cfg.seed
        self._episode_counts = np.zeros(cfg.scene.num_envs, dtype=np.int64)
        self._full_reset = False
        self._clipped_actions: np.ndarray | None = None
        self._last: BatchStep | None = None
        self._prim_paths: tuple[tuple[str, ...], ...] = ()
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, torch.Tensor], dict]:
        self._episode_counts[:] = 0
        if seed is not None:
            self._seed = seed
        self._full_reset = True
        return DirectRLEnv.reset(self, seed=seed, options=options)

    def _setup_scene(self) -> None:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)
        stage = self.scene.stage
        source = self.scene.env_prim_paths[0]
        for rod_index, n_nodes in enumerate(self._batch.spec.rod_nodes):
            _curve_with_count(stage, f"{source}/Rod_{rod_index}", n_nodes)
        self.scene.clone_environments(copy_from_source=True)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        rod_count = len(self._batch.spec.rod_nodes)
        self._prim_paths = tuple(
            tuple(f"{env_path}/Rod_{rod_index}" for rod_index in range(rod_count))
            for env_path in self.scene.env_prim_paths
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        spec = self._batch.spec
        low = np.asarray(spec.action_space.low, dtype=np.float32)
        high = np.asarray(spec.action_space.high, dtype=np.float32)
        raw = actions.detach().cpu().numpy()
        self._clipped_actions = np.clip(raw, low, high).astype(np.float32, copy=False)

    def _apply_action(self) -> None:
        if self._clipped_actions is None:
            raise RuntimeError("actions were not prepared")
        self._last = self._batch.step(self._clipped_actions)
        self._push_frame()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self._last is None:
            raise RuntimeError("reset the environment before reading observations")
        policy = torch.tensor(self._last.obs, dtype=torch.float32, device=self.device)
        return {"policy": policy}

    def _get_rewards(self) -> torch.Tensor:
        if self._last is None:
            raise RuntimeError("step the batch before reading rewards")
        return torch.tensor(self._last.reward, dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._last is None:
            raise RuntimeError("step the batch before reading dones")
        terminated = torch.tensor(
            self._last.failed, dtype=torch.bool, device=self.device
        )
        # The buffer already counts this step, so the horizon matches RodTaskEnv.
        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        ids = _as_indices(env_ids)
        if ids.size == 0:
            return
        # A full reset already zeroed the counts. Auto-reset advances only the
        # finished envs so the next stream ignores how many draws the last episode took.
        if self._full_reset:
            self._full_reset = False
        else:
            self._episode_counts[ids] += 1
        rngs = [
            _episode_generator(
                self._seed, int(index), int(self._episode_counts[index])
            )
            for index in ids
        ]
        obs = self._batch.reset(ids, rngs)
        self._store_obs(ids, obs)
        self._push_frame()

    def close(self) -> None:
        batch = getattr(self, "_batch", None)
        if batch is not None:
            batch.close()
            self._batch = None
        if not getattr(self, "_is_closed", True):
            DirectRLEnv.close(self)

    def _store_obs(self, ids: np.ndarray, obs: np.ndarray) -> None:
        obs_dim = int(self._batch.spec.observation_space.shape[0])
        if self._last is None:
            full = np.zeros((self.num_envs, obs_dim), dtype=np.float32)
            reward = np.zeros(self.num_envs, dtype=np.float32)
            failed = np.zeros(self.num_envs, dtype=np.bool_)
        else:
            full = np.array(self._last.obs, dtype=np.float32, copy=True)
            reward = self._last.reward
            failed = self._last.failed
        full[ids] = obs
        self._last = BatchStep(obs=full, reward=reward, failed=failed)

    def _push_frame(self) -> None:
        if not (self.sim.has_gui() or self.sim.has_rtx_sensors()):
            return
        origins = self.scene.env_origins.detach().cpu().numpy()
        _write_curves(self.scene.stage, self._prim_paths, self._batch.frame(), origins)
