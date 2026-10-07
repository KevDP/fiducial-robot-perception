# Real occlusion session, 2026-09-23

Thirty photographs of printed DICT_4X4_50 markers (ids 0 to 5) occluded by a dark object,
recorded to check the synthetic occlusion curve against a real capture. Each id has at least one
reference frame at zero coverage and the rest are occluded.

The photographs are not committed: they live under `data/`, like every other image in this
project. `real_occlusion_2026-09-23.csv` is the record, and it is enough to reproduce the
comparison without them.

## Columns

| Column | Meaning |
|---|---|
| `file` | Filename inside the session folder |
| `marker_id` | ArUco id |
| `occlusion_top_pct` / `occlusion_bottom_pct` | Occluder depth measured on the top and bottom black border, as a percent of marker width |
| `occlusion_mean_pct` | Mean of the two, the coverage figure used in the comparison |
| `detected_full_res` | Default `ArucoDetector` found the expected id at the photo's own resolution |
| `detected_640w` | Same, after downscaling the frame to 640 wide |
| `marker_px_at_640w` | Marker side in pixels once the frame is 640 wide |
| `tilted_edge` | Top and bottom depth differ by 5 points or more |
| `frame_only` | Occluder covers part of the white frame with white still visible beside the square |
| `paper_level` | Mean gray of the white paper, an exposure check |
| `image_width` | Pixel width of the frame as measured |

**Use `detected_640w`.** The sweep renders a marker at 96 px inside a 640 by 480 frame, so the
photographs are downscaled to the sweep's own scale before detection.

`detected_full_res` is kept because the two disagree, and the disagreement is itself a result worth having on record.

## Excluded frames

Three frames are flagged `tilted_edge`, since the generator has no tilted case they are out of the synthetic comparison.
They stay in the record because a tilted occluder is a real condition worth sweeping later.
