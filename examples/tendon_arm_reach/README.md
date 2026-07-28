# Tendon-driven continuum arm

This example ports the core environment from
[`gabotuzl/rl-cr-robot`](https://github.com/gabotuzl/rl-cr-robot) into
Gym Softrobot. The environment keeps its eight-tendon layout, target sampling,
105-value observation, action scaling, and final shaped reward.

Run a quick API and dynamics check:

```console
uv run --no-sync python examples/tendon_arm_reach/check_env.py --fast
```

Run the faithful 100-element, 800-substep configuration by omitting `--fast`.

Stable-Baselines3 is optional. After installing it, check the training pipeline
with:

```console
uv run --no-sync python examples/tendon_arm_reach/train_ppo.py \
  --fast --timesteps 4096
```

For a real experiment, omit `--fast` and increase `--timesteps`. The reference
project reports that PPO from scratch often finds degenerate policies and uses
PID demonstrations plus behavioral-cloning pretraining before PPO fine-tuning.
This example intentionally provides a minimal PPO baseline for checking the
ported environment; it does not claim to reproduce the published checkpoint.
