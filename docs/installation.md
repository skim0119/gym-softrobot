# Installation

gym-softrobot supports Python 3.12 and newer. The core install includes
Gymnasium and PyElastica. POV-Ray and interactive rendering are used only
when an environment is constructed with a matching `render_mode`.

Install the package from PyPI:

```bash
pip install gym-softrobot
```

```{note}
If a simulation imports but cannot render, the physics stack is still
installed. Rendering packages and system tools are separate from the
environment API.
```

For development, clone the repository and create the complete environment with
`uv`:

```bash
uv sync --all-groups
```
