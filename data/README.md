# data/

Generated datasets live here and are **not** committed (see `.gitignore`).

Regenerate with:

```bash
python -m fiducial.generate --out data/phase0 --seed 0
```

Every run writes a `manifest.json` next to the images. The manifest is the source of
truth for condition labels and ground truth corners; the images alone are not enough.
