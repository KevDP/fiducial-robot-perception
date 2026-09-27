# fiducial-robot-perception

Fiducial markers are the cheapest way to give an indoor mobile robot reliable landmarks,
but classical detection fails quietly in the conditions those robots actually operate in.
Robotics sits in an interesting gap between real world results and simulated ones.
This project measures where that gap is, and where it becomes challenging to close. 

## The problem

Deploying a mobile robot into an indoor space requires a complex checklist: building a map,
defining destinations, calibrating, and deciding whether the area to improve is hardware or software. 

Printed fiducial markers attack that cost directly, solving localization with a printer.
The catch is that marker detection is not the solved problem it looks like. `cv2.aruco.detectMarkers`
is a deterministic algorithm with no trained weights, and it works beautifully on a clean,
frontal, well-lit marker. Real deployments are nothing like that: people stand in front of markers, ambient light is dim and warm, windows blow out one side of the frame, and the robot sees markers off-axis.

So the real question is:

1. Under which conditions, and at what severity, does classical detection stop working, and is that gap large enough to justify a learned component at all?

Phase 0 approaches those trade-offs and sets the conditions. The question has a measurable answer:
if the classical detector holds up across the whole sweep, there is no learned component worth building 
and this project changes shape and objective.

## Results at a glance

Classical `cv2.aruco` with default parameters:

| Condition | Recall at the mildest level | Recall at the worst level |
| --- | ---: | ---: |
| Occlusion | 72.9% (5% of marker area) | **0%** (20% and above) |
| Hard shadow edge | 62.2% (edge at 0.1) | **0%** (edge at 0.3 and 0.5) |
| Motion blur | 100% (3 px kernel) | **51.1%** (15 px) |
| Backlight | 100% (0.25) | **60.0%** (1.0) |
| Low light + sensor noise | 100% (0.2) | **75.6%** (0.8) |
| Warm tint | 100% | 100% |
| Off-axis viewpoint | 100% | 100% (60 deg) |

The occlusion figure is a mean over the occluding object's tone, which is a condition
of its own. See finding 1.

These are simulator numbers, measured on the training split. What they describe is a
generator, so read them with [Limitations](#limitations).

## Important findings:

**1. Under occlusion, the tone of the object counts as much as how much it covers:**
Recall by coverage and by the tone of the object doing the covering, swept evenly
across the grey scale:

| Occluding object | 5% covered | 10% covered | 20% and above |
| --- | ---: | ---: | ---: |
| dark | 37.8% | 22.2% | 0% |
| dark grey | 40.0% | 8.9% | 0% |
| mid grey | 86.7% | 22.2% | 0% |
| light grey | 100% | 42.2% | 0% |
| light | 100% | 31.1% | 0% |

A light object at 5% costs nothing and a dark one at the same 5% costs sixty points.
Binarization sees a tone, not an object, so coverage on its own does not predict the
outcome. By 20% nothing survives at any tone.

**2. Shadow is the worst axis, and it is not a severity ramp:** Recall by edge position
across the marker:

  - 62.2% at 0.1
  - 0% at 0.3
  - 0% at 0.5
  - 4.4% at 0.7
  - 8.9% at 0.9

The curve turns back up when the marker lies almost entirely in shadow, but only by a few points.
Adaptive thresholding recalibrates against a uniformly dark region, and a shadowed print has little contrast left to recalibrate against.

**3. Failure is graded:** Where the detector still fires under occlusion or
shadow it localizes worse first: corner error runs 3 to 8 px against 0.00 px on a clean
render. Motion blur does the same. So a lost detection is preceded by a degrading one,
and the state layer has two jobs rather than one: bridging the gap, and filtering the
measurements on the way into it.

**4. The detector still fails by going quiet:** One cell out of fifty reported a wrong
id, at 2.2%. Everywhere else a failure is silence, which is the safe failure for a robot.

**Result:** warm tint and perspective up to 60 degrees do not move recall at all, so
there is no case for building "general robustness".
The targets of the learned layer are partial occlusion and strong local gradients, the
two axes that fail hardest and earliest.

## Architecture

| Layer | Role | Status |
| --- | --- | --- |
| Classical | `cv2.aruco`, default parameters, as the baseline to beat | phase 0 |
| Learned | small detector to recover what the classical one loses | phase 1 |
| State | Kalman filter holding pose through lost detections | phase 2 |

The state layer comes from [kalman-filter-tracker](https://github.com/KevDP/kalman-filter-tracker),
where it was already validated on synthetic and real video. This matters because it decouples the detector's
frame rate from the control loop, so perception can run at a few frames per second and still publish smooth pose.
This explains why it keeps a small CPU-only model viable on embedded hardware.

## Key technical decisions

**One degradation axis per sample**
Crossing axes confounds attribution. Combined conditions are a later phase, once the per-axis curves exist.

**Level 0.0 is generated through the same code path as every degraded level.**
The clean sample and the undegraded end of each curve are then bit-identical by construction.

**Splits are over sequences.**
Every degraded sample derived from one scene shares its background, placement and marker id. Splitting at the sample level puts near-duplicates on both sides and inflates every downstream number. See `tests/test_splits.py`.

**The dataset is reproducible, so the seal is auditable without the images.**
Running `python -m fiducial.generate --seed 0` checks that the manifest fingerprint matches the committed seal. The fingerprint covers what the seed decides: the scene parameters and the split.

**The holdout is sealed and every access is logged.**
`experiments/split.sealed.json` is fingerprinted against the manifest it was computed from. 
Regenerating the dataset invalidates it instead of silently reusing it. Evaluating against the holdout requires an explicit flag and a stated reason, and appends to `experiments/holdout_access.log`. 

**Low light adds sensor noise.**
Scaling brightness down is recoverable by any contrast normalization. A real camera raises gain in a dim
room, and the noise that comes with it is the part that actually needs solving.

## Limitations

- **The printed appearance is drawn from chosen ranges** Ink,
  paper and the page around the marker are randomized per sequence, over ranges wide enough
  to hold prints that a reader would call plausible. That makes a curve describe behaviour
  across printouts, but the ranges are a judgement and not a measurement.
- **A flat curve is ambiguous.** Warm tint and off-axis viewpoint not breaking anything is
  consistent with ArUco being robust, and equally consistent with the synthetic degradation
  being too gentle.
- **One occluder geometry.** The occluding object enters from an edge and covers everything
  behind it, with a hard edge and casting no shadow of its own on the page.
- **The degradations are synthetic.** Degradations can only approximate real failure modes. A real occlusion 
  has texture and a real dim frame has sensor-specific noise.
- **Backgrounds are not photographic.** In real clutter there are contours that this generator does not reproduce.
- **One marker per frame.** At least in phase 0, multi-marker scenes are out of scope.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # .venv/Scripts/activate on Windows
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
    detect.py         The classical ArUco detector
    metrics.py        Recall, wrong-id rate and corner RMSE, per condition
    splits.py         Sequence-level split, sealing, and the holdout access log
    generate.py       Entry point: build the dataset and seal the split
    baseline.py       Entry point: measure the classical detector

tests/                One test_<module>.py per module, written as invariant guards
experiments/          Sealed split, access log and experiment records
data/                 Generated images and manifests
markers/              Printable marker sheets
```
