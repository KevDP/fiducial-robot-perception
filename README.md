# fiducial-robot-perception

Fiducial markers are the cheapest way to give an indoor mobile robot reliable landmarks,
but classical detection fails quietly in the conditions those robot actually operate in.
Robotics have an interesting situation between real world and simulated data results.
This project measures the case and where it can be challenguing to provide solutions. 

## The problem

Deploying a mobile robot into an indoor space requires a complex checklist: building a map,
defining destinations, calibrating, whether area to really improve between hw or sw. 

Using printed fiducial markers attack that cost directly, solving localization with a printer.
The catch is that marker detection is not the solved problem it looks like. `cv2.aruco.detectMarkers`
is a deterministic algorithm with no trained weights, and it works beautifully on a clean,
frontal, well-lit marker. Real world deployments are none quite of this performance: people stand in front of markers, ambient light is dim and warm, windows blow out one side of the frame, and the robot
sees markers off-axis.

So the real question it is:

1. Under which conditions, and at what severity, does classical detection stop working, and is that gap large enough to justify a learned component at all?

Phase 0 is approaching all the trade-offs and setting conditions. Previous question has a measurable answer,
If the classical detector holds up across the whole sweep, there is no learned component worth building 
and this project will change shape and objective.

## Results at a glance

Classical `cv2.aruco` with default parameters:

| Condition | Recall holds until | Recall at worst level |
| --- | --- | ---: |
| Occlusion | 20% of marker area | **0%** (at 40% and above) |
| Hard shadow edge | edge at 10% across | **4.4%** (edge near the midpoint) |
| Motion blur | 11 px kernel | **40.0%** (at 15 px) |
| Low light + sensor noise | never breaks | 100% |
| Backlight | never breaks | 100% |
| Warm tint | never breaks | 100% |
| Off-axis viewpoint | never breaks (60 deg) | 100% |

## Important findings:

**1. Occlusion has a cliff:** Recall is 100% at 20% coverage and 6.7% at 30%.
With this scale, there is no gradual band to lean on.

**2. Shadow difficulty is not monotonic.** Recall by edge position across the marker: 

  - 100% at 0.1
  - 13.3% at 0.3
  - 4.4% at 0.5
  - 62.2% at 0.7
  - 100% at 0.9. 
  
A marker lying almost entirely in shadow is recovered fine, one split down the middle is destroyed.
Adaptive thresholding recalibrates against a uniformly dark region, but a strong gradient breaks 
binarization on half the modules.

**3. Failure is binary, not graded.** Only motion blur produces genuine localization degradation.
Everywhere else the detector either nails it or reports nothing, which is precisely why a state layer
is needed. Wrong-id rate was 0.0% in every cell.

**Result:** Global photometric conditions (dim light with sensor noise, backlight, warm tint) 
and perspective up to 60 degrees do not move recall at all, so there is no case for building 
"general robustness". The targets of the learned layer are partial occlusion and strong local gradients.

## Architecture

| Layer | Role | Status |
| --- | --- | --- |
| Classical | `cv2.aruco`, default parameters, as the baseline to beat | phase 0 |
| Learned | small detector expected to recover the results that classical measure losses | phase 1 |
| State | Kalman filter holding pose through lost detections | phase 2 |

The state layer comes from [kalman-filter-tracker](https://github.com/KevDP/kalman-filter-tracker),
where it was already validated on synthetic and real video. This matters because it decouples the detector's
frame rate, so perception can run at a few frames per second and still publish smooth pose.
This explains why it keeps a small CPU-only model viable on embedded hardware.

## Key technical decisions

**One degradation axis per sample**
Crossing axes confounds attribution. Combined conditions are a later phase, once the per-axis curves exist.

**Level 0.0 is generated through the same code path as every degraded level.**
The clean sample and the undegraded end of each curve are then bit-identical by construction.

**Splits are over sequences.**
Every degraded sample derived from one scene shares its background, placement and marker id. Splitting at the sample level puts near-duplicates on both sides and inflates every downstream number. See `tests/test_splits.py`.

**The dataset is reproducible, so the seal is auditable without the images.**
By running `python -m fiducial.generate --seed 0` it is possible to check that the manifest fingerprint matches the committed seal. This only holds because generation is deterministic across processes.

**The holdout is sealed and every access is logged.**
`experiments/split.sealed.json` is fingerprinted against the manifest it was computed from. 
Regenerating the dataset invalidates it instead of silently reusing it. Evaluating against the holdout requires an explicit flag and a stated reason, and appends to `experiments/holdout_access.log`. 

**Low light adds sensor noise.**
Scaling brightness down is recoverable by any contrast normalization. A real camera raises gain in a dim
room, and the noise that comes with it is what actually needs to be approached with real solutions.

## Limitations

- **A flat curve is ambiguous.** The results are unconsistent with ArUco regarding the challenguing way to
  handle global photometric change, is expected that synthetic dim-light conditions did not break it.
- **The degradations are synthetic.** Degradations only can approximate real failure modes. A real occlusion 
  has texture and a real dim frame has sensor-specific noise. This requires a validation work step.
- **Backgrounds are not photographic.** In real clutter there are contours that this generator does not reproduce.
- **One marker per frame.** At least in phase 0, a multi-marker scene are out of scope.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Generate the sweep and seal the split (one time):

```bash
python -m fiducial.generate --out data/phase0 --seed 0
```

Measure the classical baseline on the training split:

```bash
python -m fiducial.baseline --dataset data/phase0 --out experiments/baseline_aruco.json
```

The holdout stays closed until the end, and opening it is deliberate:

```bash
python -m fiducial.baseline --holdout --reason "final phase-0 baseline"
```

Run the invariant guards:

```bash
make test
make lint
```

## Repository structure

```
src/fiducial/
    config.py         Shared constants: image geometry, marker family, the sweep
    scene.py          Renders clean marker scenes on textured backgrounds
    degrade.py        The seven degradation axes, one function each
    imagestats.py     Objective per-sample statistics recorded in the manifest
    dataset.py        Generation and manifest handling
    detect.py         Classical ArUco baseline
    metrics.py        Recall, wrong-id rate and corner RMSE, per condition
    splits.py         Sequence-level split, sealing, and the holdout access log
    generate.py       Entry point: build the dataset and seal the split
    baseline.py       Entry point: measure the classical detector

tests/                One test_<module>.py per module, written as invariant guards
experiments/          Sealed split, access log and experiment records
data/                 Generated images and manifests
```
