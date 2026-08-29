# SATELLITE

**[Read the documentation](https://tristangiando.github.io/satellite/)** — three parts:

| Part | Page | What it covers |
| --- | --- | --- |
| I | [Results](https://tristangiando.github.io/satellite/) | the images |
| II | [Research](https://tristangiando.github.io/satellite/research.html) | training, cloud masks, compositing, residual failures |
| III | [Engineering](https://tristangiando.github.io/satellite/engineering.html) | pipeline, profiling, architecture |

```bash
uv sync
uv run python satellite/scripts/run_inference.py   # one tile, one month
uv run python satellite/scripts/run_year.py        # twelve mosaics, one per month
uv run python satellite/scripts/regrade_mosaics.py # re-grade the stored mosaics
```
