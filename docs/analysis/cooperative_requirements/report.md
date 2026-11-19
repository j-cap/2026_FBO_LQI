# Requirement sensitivity: common weights versus donor calibration

No new optimization or simulation. All thresholds below are illustrative, not externally justified requirements.

## Result

The retrospective common weights have worst per-maneuver mean measured RMSE
**0.911601 deg/s** across 6 vehicles x 5 bandwidths x 2 maneuvers.
The target-specific references have worst mean **0.911985 deg/s**.
References were optimized for the combined measured objective; neither reference
nor common weights establish the minimax optimum or an infeasibility bound.

At 0.92 deg/s, common weights satisfy all 60 condition/maneuver sample means.
The donor comparison uses the SAME five-other-vehicle target coverage for both
common and donor weights. Full donor-inclusive calibration success remains unknown.

| Threshold (deg/s) | BO25 | BO75 | Sobol25 | Sobol75 | Common full conditions |
|---|---:|---:|---:|---:|---:|
| 0.90 | 0/18 | 0/18 | 0/18 | 0/18 | 58/60 |
| 0.91 | 1/18 | 2/18 | 1/18 | 2/18 | 59/60 |
| 0.92 | 9/18 | 14/18 | 9/18 | 9/18 | 60/60 |
| 0.93 | 11/18 | 16/18 | 14/18 | 18/18 | 60/60 |
| 0.94 | 17/18 | 18/18 | 14/18 | 18/18 | 60/60 |
| 0.95 | 18/18 | 18/18 | 18/18 | 18/18 | 60/60 |

## Why 0.92 deg/s is not yet a defensible requirement

The worst common-weight case is seed 987946, tire_degraded_00,
tau=0.4 s, maneuver=1 s.
Its 40 episode RMSEs have standard deviation 0.0616 deg/s.
An exploratory pointwise percentile bootstrap interval for the mean is
**[0.8931, 0.9310] deg/s**. The 0.92 threshold lies inside it.
This interval ignores selection of the worst condition and is not a simultaneous
fleet guarantee. Episodes provide noise replication, not independent fleet replication.

The paired common-minus-reference difference on this condition is
-0.000384 deg/s, with exploratory interval
[-0.004314, 0.004602] deg/s.
Similar common/reference performance does not remove uncertainty about an absolute limit.

## Corrected interpretation of the demanding condition

Payload/tau=0.4 often has the largest cost over the FULL screening grid.
For the good common weights, tire_degraded/tau=0.4 and the faster maneuver limit
performance. A grid-wide worst-case frequency is therefore not enough to choose
one representative vehicle for calibration. Any demanding-condition rule selected
from these results is exploratory and needs independent confirmation.

## Requirement specification to complete before Stage B

1. Meaning: expected episode yaw-rate RMSE separately for each prescribed maneuver,
   or a specified probability of exceeding a limit? These are different requirements.
2. Signal: measured RMSE includes sensor noise; true-state RMSE is currently only a
   simulation diagnostic. Use a common measurement convention for any deployment claim.
3. Value: supply an absolute limit from an engineering acceptance rule or partner
   specification. Alternatively declare an illustrative benchmark requirement explicitly.
4. Evidence: specify accepted uncertainty and physical-constraint violation tolerance.
   Sample means below the limit alone are not a reliability certificate.
5. Scope: finite evaluated vehicles and maneuvers initially; unseen-vehicle claims need
   independent fleets. Preserve model-specific gain redesign and the physical limits.

## Next decision

Do not launch fleet BO based on the 0.92 threshold alone. First settle the acceptance
meaning and engineering limit. If a tight limit is justified, use fresh noise episodes
to validate frozen common/reference candidates before spending on optimization.
Use a sample-size plan and one fixed final analysis, or a sequentially valid rule;
do not repeatedly sample until an ordinary confidence interval passes.
If no practical limit is available, this can only be an exploratory benchmark study.
The existing 0.92/0.93/0.95 sensitivity grid must not be sold as a preregistered
engineering requirement. No current result demonstrates experiment-count savings.
