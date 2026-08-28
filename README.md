# SATELLITE

Cloud-free Sentinel-2 composites: a U-Net segments cloud on every pass over a tile,
and a compositor keeps, for each pixel, the single cleanest observation of the month.

**[Read the working log →](https://tristangiando.github.io/satellite/)**

```bash
uv sync
uv run python satellite/scripts/run_inference.py
```

The documentation is a static site in [`docs/`](docs/) — plain HTML, CSS and JS, no build step.
Pushing to `main` publishes it to GitHub Pages.
