# Trajectory-tracking diagnostic

Prepared 23 November 2026. Branch: `study/trajectory-tracking-diagnostic`.
Configuration: `code/config/runs/tracking_diagnostic.json`.

## Purpose and decisions

Before building another BO variant, determine whether the trajectory-tracking
task supplies useful controller-tuning headroom, reference dependence, and donor
structure. These are finite-pool simulation diagnostics. The models are synthetic
CFL-FF anchors, not validated parameter distributions for production vehicles.

| Question | Controlled comparison | Evidence to inspect |
|---|---|---|
| Q1: Does tuning help, and does each vehicle need different gains? | Original gains, one common scaling, one common free three-gain choice, one fixed choice per vehicle | Independent validation cost, tracking/steering components, coverage, and paired differences |
| Q2: Does the useful gain choice depend on speed and reference dynamics? | Vehicle-fixed versus vehicle-and-context gains; near-optimal sets across contexts | Validation cost, gain changes, 5% near-optimal-set Jaccard, unavailable/empty sets |
| Q3: Can a declared similarity measure select useful source gains? | Same-context donor transfer using family, physical parameters, DC yaw gain, DC plus transient time constant, and reference-weighted closed-loop response | Target validation cost and failures, uniform-donor reference, within-target donor ranks |

The practical reading order is Q1, Q2, Q3. Improvement over the original hand
tuning alone does not establish a need for fleet knowledge. If a common scaled
controller performs as well as individualized gains, the simpler calibration is
adequate here. If vehicle-fixed and context-specific gains give similar costs and
overlapping useful sets, this task does not support a contextual method. If the
oracle control-informed distance cannot choose useful donors, do not proceed by
adding an estimator to rescue this particular similarity claim.

Conversely, a useful oracle donor relation motivates a separate **limited-budget
BO experiment** against strong independent BO, common-controller tuning, and
unfiltered transfer. It does not itself establish an information limitation,
sample-efficiency benefit, a federated algorithm, or a deployable similarity
estimator. Report relative differences and individual cases without treating the
nine deterministic plants or dependent donor pairs as population samples.

## Dynamic models and controller

Use one six-state nonlinear bicycle implementation throughout:
`[X, Y, psi, vx, vy, r]`, lateral Pacejka forces, static axle loads, direct
longitudinal force requests, and per-axle friction circles. There is no additional
actuator lag, delay, process noise, load-transfer model, or random manufacturing
scatter in this diagnostic. The selected original CFL-FF scalar source is retained
under `code/src/tracking_reference/`; the vectorized implementation is tested
against it, including combined-force saturation and low/reverse-speed guards.

| Family | Mass [kg] | Yaw inertia [kg m²] | lf / lr [m] | Front B,C,E | Rear B,C,E |
|---|---:|---:|---|---|---|
| compact / mini_D | 1220 | 1500 | 1.05 / 1.45 | 9, 1.27, .93 | 10, 1.27, .93 |
| sport / porsch911_E | 1520 | 1800 | 1.15 / 1.35 | 13.5, 1.38, .86 | 14, 1.38, .86 |
| suv / heavySUV_F | 2550 | 5200 | 1.30 / 1.55 | 7.5, 1.22, .95 | 8, 1.22, .95 |

For each anchor, use three separate configurations: nominal, a point payload of
20% of the original mass at `d = .15 L` rearward from its original CG, and a
front-tire intervention `Bf *= .85` with the rear tire unchanged. Payload and tire
interventions are not combined. Common friction coefficient is `.9`.

For payload, `m_new = m + mp`, `shift = mp*d/(m+mp)`, `lf_new = lf + shift`,
`lr_new = lr - shift`, and `Iz_new = Iz + m*mp/(m+mp)*d²`. The latter is inertia
about the **new CG**, correcting the archived simplified `Iz + mp*d²` formula.
No intrinsic payload yaw inertia is modeled. Use numeric anchor geometry: the
archived sport wheelbase comment is inconsistent with its `lf+lr = 2.50 m`.
Archived longitudinal Pacejka parameters are inactive in the corrected force
law and are not varied or used as physical similarity features.

The desired-frame position/heading errors use desired minus actual state:

`delta_fb_raw = Ky*ey + Kpsi*wrap(psi_ref-psi) + Kr*(r_ref-r)`.

Preserve the CFL clipping order: clip feedback alone to ±12°, add
`atan(L*kappa_ref)`, clip total angle to ±12°, then apply a 300°/s rate limiter.
At 50 ms, the rate increment is 15°; crossing the full ±12° range still requires
the limiter. The longitudinal controller remains
`ax = clip(1.0*ex + .1*(v_ref-vx), -6, 3)` in m/s². The plant additionally has
its original acceleration clamp and friction-circle coupling. No neural
feedforward model is used. Original lateral gains are `[.005, .1, .1]`.
The study tunes these three feedback gains, not LQI Q/R matrices.

## References and split

Six controlled contexts cross constant speed `{20, 30} m/s` with carrier frequency
`{.1, .3, .6} Hz`. Episodes last 25 s at a 50 ms controller period. A C2 smoothstep
window starts at 2 s, ends at 22 s, and uses two-second ramps. Its sinusoid is
normalized to unit peak on the controller grid. Set
`kappa_ref = 2.5/v² * shape`, `r_ref = v*kappa_ref`.

All six contexts have the same sampled peak reference lateral demand 2.5 m/s².
The 20/30 m/s pairs deliberately have different geometric curvature. This isolates
reference timing at matched demand; it is not the same road driven faster.
Heading integrates the cubic yaw-rate interpolant analytically. Position uses
cubic antiderivatives on a 10x finer grid, tested against an analytic circle.
Initial conditions match the initial straight reference with zero sideslip.
Actual body acceleration `ay = dot(vy) + vx*r` is logged separately from `v²*kappa`.

The held-out bank contains **15 references**, fixed before plant evaluation:

- Single lane change, double lane change, and smooth curve at 20, 25, 30 m/s.
  Lane-change curvature uses the original CFL-FF quintic helpers, a 3.5 m lane,
  a 4 s single maneuver, and 1 s per half for the double. Start is 4 s. The curve
  uses a 12 s sine-squared curvature pulse with peak .0015 1/m. The original
  6/7 m/s² lane-change demand caps, .07 1/m curvature cap and .08 1/(m s)
  curvature-slew cap apply. Clipped lane changes need not preserve exactly 3.5 m.
- Two original-generator trajectories per speed, seeds 3101/3102. All generator
  settings are explicit copies of the archived `default` preset; only its speed
  choices are restricted to one constant speed. Recompute pose with the accurate
  integrator. Do not reject, regenerate, or replace draws based on their events,
  tracking cost, or any plant outcome.

The guaranteed maneuver set prevents random draws from accidentally omitting a
maneuver type. The generator draws retain the original mixture. Held-out cases
at 20/30 m/s test new shapes; 25 m/s tests interior speed generalization as well
as new shapes. They also change peak demand, so they do not identify an isolated
frequency effect. There is no retuning on this bank.

The context-specific policy maps a held-out reference to a controlled prototype
by minimizing `((v-vq)/10)² + log(f_centroid/fq_centroid)²`; ties use context ID.
Centroids use the full prescribed reference curvature FFT. Donor-local gains
use the same prototype assignment, while oracle distances can use the actual
held-out reference spectrum. This uses known desired references, not target
tracking outcomes. It does not test a learned or interpolated gain schedule.

## Gain pool, objective, and feasibility

Every controlled cell evaluates the **same 71 candidates**: 64 scrambled Sobol
points, seed 731, in independent log10 gain multipliers `[-1,1]^3`, plus common
scalings `{1,.5,.8,1.2,2,4,8}` of the original gains. Candidate 0 is the original
controller. These are finite-pool minima, not global optima. The report flags
candidates within 10% of the log-domain width from any boundary. Boundary
selection is an unresolved search-range issue, not evidence of a localized optimum.

Fix the score before the run:

`J = RMSE(true ey)/(0.25 m) + RMS(applied steering rate)/(10 deg/s)`.

Compute lateral RMSE at all 501 state sample times, and steering rate from the
500 applied-control differences, including the initial zero steering. Both terms
cover the full 25 s episode, including initial straight and recovery. Scales and
equal normalized weights are **research choices**, not comfort specifications or
calibrated engineering requirements. Heading, longitudinal tracking, steering
RMS, acceleration peaks, tire utilization, and clipping fractions are diagnostics.
The score uses truth; feedback uses noisy measurements.

Eligibility requires completion, numerical resolution, peak true `|ey| <= 2 m`,
`|beta| <= 6°`, and speed error `<= 2 m/s`. Check peaks at controller samples and
integration substeps with linearly interpolated desired pose for the substep
error monitor. These are developmental task criteria, not certified safety bounds.
Exceeding a task criterion is retained as failure while stable simulation continues
to the end. Original generous divergence guards (nonfinite state/derivative,
40° sideslip, reverse speed below -2 m/s, or yaw rate above 5 rad/s) quarantine the
candidate; its partial score is unavailable. Clipping itself is logged and is not
an automatic failure. No shortened or infeasible rollout enters a best-gain choice.

Noise is independent Gaussian measurement noise on `[X,Y,psi,vx,vy,r]`, with
standard deviations `[.02 m,.02 m,.001 rad,.05 m/s,0,.001 rad/s]` from CFL-FF.
There is no sideslip sensor or new process noise. The original ay sensor noise
is irrelevant to this controller and is not synthesized. Calibration seeds are
101/102; validation seeds are 201/202. Common random numbers are paired across
candidates, vehicles, and integration resolutions within each reference/seed;
different reference IDs have independent streams. Seeds are hashed stably.
Different noise seeds are repeated trials, not different desired trajectories.

RK4 holds each sampled control for the full controller period and adaptively
refines only unresolved candidates through 1, 2, 4, and at most 8 substeps.
Consecutive resolutions must agree in
completion/failure classification, lateral RMSE, steering-rate RMS, and peaks
of lateral error, sideslip, speed error, and actual ay. Component tolerance is
`1e-5 + 1e-3*max(abs(fine_component),1e-3)` in the reported component's units.
Compare all quantities in this condition, not just aggregate J. Two matching
divergence failures remain ineligible regardless of resolution. Unresolved
trials are retained, flagged, and excluded from selection.

## Selection and transfer analysis

For each gain, take mean J over both calibration seeds, provided **both** seeds
are eligible. Choose one common gain minimizing the uniform 54-cell mean subject
to eligibility everywhere; repeat within the seven common scalings. For a
vehicle-fixed gain, average its six contexts with equal weights. For a context
gain, minimize within that cell. Missing feasible choices remain unavailable;
there is no outcome-dependent fallback.

Freeze all selected candidate IDs before either validation bank is simulated. Controlled validation
uses the same six prescribed contexts with independent measurement noise. A
policy's fleet mean is unavailable if any cell fails; conditional means and
coverage are reported separately. Do not compare conditional means across
unequal coverage as a claim of improvement. The summary also computes paired
common-versus-context differences only on jointly eligible cells, with its count.

Q2 near-optimal sets contain eligible calibration candidates within 5% of that
cell's minimum. Report intersection and Jaccard for every within-vehicle context
pair. Empty sets are marked unavailable, not identical.

For Q3, each donor contributes its own calibration-selected gain for that context.
Never include the target itself. Evaluate donor gains on the target's validation
seeds. A gap to the target's calibration-selected gain can be negative under
validation noise; do not clamp it. Rank donors using these predefined distances:

| Distance | Definition | Information assumption |
|---|---|---|
| Family | 0 for same family, 1 otherwise | Known family label |
| Physical | Euclidean distance in `[m,Iz,lf,lr,Bf,Cf,Ef,Br,Cr,Er]` divided by `[1500,2500,1.2,1.4,10,1.3,1,10,1.3,1]` | Exact parameters, oracle |
| Steady | Absolute log ratio of DC yaw gain normalized by `v/L` | Exact tangent at that speed, oracle |
| Transient | Euclidean norm of log normalized-DC ratio and log dominant open-lateral time-constant ratio | Exact tangent at that speed, oracle; a two-feature descriptor |
| Closed loop | Reference-spectrum-weighted RMS difference of baseline lateral-error and steering-rate transfer responses, normalized by the objective scales | Exact tangent and original fixed feedback gains, oracle |

For the closed-loop descriptor, linearize the true nonlinear `[vy,r]` dynamics
at straight driving. Add desired-frame states `[ey,e_psi]` with
`dot(ey) = v*e_psi - vy`, `dot(e_psi)=v*kappa-r`. Apply exact ZOH discretization
at 50 ms and then sampled feedback. Use the full reference FFT, correct real-signal
Parseval weights, and the discrete applied-steering difference operator. This
descriptor ignores active clips and nonlinear departures from straight driving.
It uses a held-curvature disturbance approximation; actual reference pose is
integrated smoothly. It is a frequency-domain similarity proxy, not a validated
prediction of each nonlinear finite-horizon tracking rollout.
Unavailable/unstable descriptors have no donor choice; no target costs repair them.
All distances and scales are fixed without fitting target outcomes. Ties use donor
ID, including the deliberately coarse family comparator.

Archive every donor-target pair, including failed transfers, and a uniform random
donor reference with feasibility probability. Expected J is unavailable if any
eligible source donor fails on that target; a conditional mean is labeled as such.
Spearman correlations are computed **within each target/context**, then summarized
descriptively. There are no pooled-pair p-values or independent-sample claims.
No deployment identification, communication protocol, privacy guarantee, or
federated optimizer is implemented at this stage.

## Run locally

The existing `paper_FBO_LQI` environment is sufficient; no new simulation package
is needed. On Windows, from the repository root:

```powershell
git fetch origin
git switch study/trajectory-tracking-diagnostic
git pull --ff-only origin study/trajectory-tracking-diagnostic
conda activate paper_FBO_LQI
python -m unittest discover -s code/tests -p test_tracking_diagnostic.py -v
python code/scripts/tracking_diagnostic.py preflight
python code/scripts/tracking_diagnostic.py smoke --workers 4
python code/scripts/tracking_diagnostic.py run --workers 4
start code\results\tracking_diagnostic_v1\report.html
```

The full controlled bank has 216 vectorized batches / 15,336 gain episodes.
The realism bank has 270 batches evaluating only the distinct gains selected for
that vehicle/reference. Each episode may be integrated at multiple resolutions.
Four workers run independent batches; each batch vectorizes candidates. The smoke
run uses three configurations spanning all families/interventions, two controlled
contexts, 11 candidates, one noise seed per role, and 12 realism references. It is
a software check and cannot answer the scientific questions.

Every completed batch checkpoints atomically. Repeating `run` resumes completed
batches; `status` reports counts. Code/configuration fingerprints ignore CRLF
differences and exclude the git commit ID, so committing the same files does not
invalidate a run. Changing simulation/selection code or configuration requires a
new result tag; old results are never overwritten or mixed. Runtime versions,
git head, dirty-worktree status, original-source provenance, actual references,
selected gains, and the held-out evaluation schedule are recorded.

```powershell
python code/scripts/tracking_diagnostic.py status
python code/scripts/tracking_diagnostic.py analyze
python code/scripts/tracking_diagnostic.py collect
git status --short
git commit -m "Add trajectory tracking diagnostic results"
git push origin study/trajectory-tracking-diagnostic
```

`collect` revalidates checkpoint identity, candidate completeness, source signature,
and completion of both controlled and realism banks before staging **only** the
full run's result directory. It refuses smoke collection. Inspect any previously
staged unrelated changes before committing. Reports are self-contained HTML plus
CSV/JSON/PNG and compressed reference/representative selected-policy traces.
Legacy yaw-rate runs and manuscripts are not part of this experiment's outputs.
