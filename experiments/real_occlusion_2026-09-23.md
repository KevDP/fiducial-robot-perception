# Real occlusion session, 2026-09-23

Thirty photographs of printed DICT_4X4_50 markers (ids 0 to 5) occluded by a dark object,
recorded to check the synthetic occlusion curve against a real capture. One reference frame
per id at zero coverage plus four occluded frames.

The photographs are not committed: they live under `data/`, like every other image in this
project. `real_occlusion_2026-09-23.csv` is the record, and it is enough to reproduce the
comparison without them.

## Capture

- Samsung Galaxy S25 Ultra: 1x lens, 4:3, ISO 250, 1/10 s, tripod, timer.
- Printed sheets from `markers/`: 150 mm black square, 18.75 mm white frame, US Letter.
- Occluder: dark green book cover entering from the right edge, measured at about 55 in gray.
- Printed ink measures about 57 and the paper about 170 under this exposure, against the 0 and 255 the generator renders.

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

**Use `detected_640w`.** The sweep renders a marker at 96 px inside a 640 by 480 frame. 

These photographs are 2576 wide, where the marker spans close to 390 px, the frames are downscaled to the sweep's own scale.
At that scale the marker measures 96.9 to 98.1 px in the unoccluded frames, which confirms the framing was set correctly.

`detected_full_res` is kept because the two disagree, and the disagreement is itself a result worth having on record.

## Excluded frames

Three frames are flagged `tilted_edge`, since the generator has no tilted case they are out of the synthetic comparison.
They stay in the record because a tilted occluder is a real condition worth sweeping later.
