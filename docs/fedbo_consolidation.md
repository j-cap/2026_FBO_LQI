# FedBO: consolidated findings and the bigger picture

25 November 2026. This supersedes the
[24 November consolidation](fedbo_consolidation_2026_11_24.md), which remains
archived unchanged. Latest evidence: the
[three-study six-speed audit](analysis/six_speed_consolidation/report.md).

Subsequent [gain-domain audit](analysis/gain_domain/report.md): denser sampling
and a genuine K_y extension improve the offline reference, especially at
0.1 Hz. The online results below retain their original 71-vector task and
reference. Their success rates are not recomputed against controllers that
were outside their permitted action set.

**Online fleet cost information helps the declared near-reference success
metric in the tested homogeneous task. It has not yet produced a method that
dominates the practical references on cost, failed experiments and complete
calibration together.** The beta = 0 / 0.5 comparison identifies a meaningful
exploration tradeoff. Further bonus tuning is not the next scientific question.

## The problem and method we actually have

The current code tunes three feedback gains, Ky, Kpsi and Kr, of a fixed
trajectory-tracking controller with kinematic steering feedforward. The
plant is the nonlinear Pacejka bicycle inherited from CFL-FF. The legacy
repository title notwithstanding, the current task directly tunes these
gains, rather than LQI Q/R weights or a learned inverse feedforward network.

The physical objective J is lateral-tracking RMSE divided by 0.25 m plus
applied-steering-rate RMS divided by 10 deg/s. Gains are selected from a
public 71-vector pool. A trial holds one assigned speed/frequency context
and gain vector for a 25-second closed-loop rollout. Context is externally
assigned and known before gain selection. Learning is online **between
trials**; the experiment does not adapt gains within a rollout.

A vehicle pays for one factory test and at most 23 subsequent tests. There
is no historical cost archive at initialization. Each arm owns its paid
history and peer catalogue, and communicates after all round actions and
observations. Failed tests consume budget.

D learns a Bayesian contextual log-cost model with 257 public spectral
coefficients, a shared component and a client-specific component. Inputs
are three gain coordinates, speed and log maneuver frequency. An evidence
mixture over sharing strengths learns how much the cost surfaces should
couple. E has the same marginal prior with independent cost models. A uses
an adaptive local GP; B adds a locally fitted 65-coefficient RFF mean.
**All four methods use the same paid, exact-context peer-first rule.** E
and A/B therefore already benefit from the peer-gain channel.

Current D training uses analytical Gaussian information messages, not FedAvg
or stochastic gradient training. The server combines precision/information
statistics and evidence scalars; raw trajectory records are not inputs to
that server interface. This is a federated statistical reference in a
simulation, with exact inference for the stated model. It establishes no
formal privacy guarantee or deployed network performance.

## What the experiment chain now establishes

| Question | Current conclusion | Relevant evidence |
|---|---|---|
| Is there useful tuning headroom? | Yes; controller quality depends strongly on reference context | Dense tracking and historical-context diagnostics |
| Can fleet experience be generated online? | Yes; all new tests are charged and histories start empty | One-factory-test online protocol and subsequent studies |
| Does a cost model add information beyond directly transferring gains? | Yes for the homogeneous primary success metric in the tested model class | D/E with identical marginal priors, acquisition and paid peer rules |
| Does every federated architecture help? | No; the earlier C/B heterogeneous result deteriorated | Frozen matched FedAvg-mean comparison |
| Does the spectral D formulation have evidence beyond the pilots? | Yes, in the earlier two-speed confirmation, within its own protocol | Homogeneous D/E succeeds under its declared criterion; heterogeneous D/E remains unconfirmed |
| Does adding more speeds automatically create missing local speed ranges? | No; clients still encounter about 5.9 of six speeds | Six-speed programme coverage |
| Is the explicit cost bonus unnecessary? | Zero improves some outcomes, but repeatedly misses 0.3 Hz discoveries; beta = 0.5 recovers them with more failures | Matched exploration comparisons |
| Have we beaten A and then B? | On the latest declared success metric, yes in all three seeds; across all practical metrics, no | D0.5/A/B, paired costs, failed tests and complete-policy counts |
| Is there a communication result? | No; the exact reference is expensive and its cost is now quantified | About 660 MB of model payload per fleet through the final decision |
| Are model messages established as preferable to sharing trial summaries? | No; the cost label is already a scalar and no matched observation-sharing policy is tested | The current comparison establishes the value of additional cost information, not the necessity of this exchange mechanism |

The historical work established transfer potential, but its archive was
already available. The online work resolves that accounting issue. Matching
the peer rule resolves a second ambiguity: D/E measures what sharing the
cost-model information adds beyond an existing controller catalogue.

The architectural changes also matter. The earlier extra FedAvg mean C
helped homogeneous clients and hurt the heterogeneous comparison. Joint
shared/local inference changed that result, and spectral features improved
the representation. Those findings do not imply that conventional FedAvg
alone solves the current task, or that every form of sharing is beneficial.
The [earlier consolidation](fedbo_consolidation_2026_11_24.md) preserves that
history and its narrower claims.

## The latest result and the exploration tradeoff

In the new matched study, D with beta = 0.5 achieves 54.66% primary success,
versus A's 38.71%, B's 27.02%, matched E's 42.55% and zero-bonus D's 44.18%.
Its advantages over A, B, E0.5 and D0 are positive in all three seeds.
At 0.3 Hz it recovers 43.40% success from D0's zero; useful discoveries occur
in all three seeds. This is progress on the stated primary objective.

However, D0.5 incurs 78 failed tests versus D0's 46 and E0.5's 50. It has
14 complete eligible policies, versus D0's 15, B's 20 and E0.5's 19.
On the common complete-policy subsets its mean J is 29.33% above D0's and
1.71% above A's. Its predeclared clean-improvement criterion fails. These
are objective and coverage tradeoffs, not an invalid experiment.

D0 and D0.5 should remain separate references: D0 for its lower observed
failure burden and lower paired mean cost in this comparison; D0.5 for its
higher near-reference hit rate and recovered discoveries. Neither merits a
claim of uniform superiority. Changing the success threshold after the fact
would not resolve the underlying choice of operational objective. A
post-hoc threshold check preserves D0.5's hit-rate advantage at 2.5%, 10%
and 20% too, but leaves its failure and cost tradeoffs intact.

The two zero-bonus studies repeat the lack of 0.3 Hz discoveries across six
programme seeds. Conversely, D/E primary-success effects remain positive in
aggregate under beta = 2, 0 and 0.5 in their respective matched studies.
This supports useful shared cost information while showing that discovery
and eligibility must be assessed separately. The three studies are
exploratory and adaptively sequenced; they are not a pooled confirmation.

## Where the practical limitation lies

All current six-speed clients have the same plant and speed distribution.
They encounter roughly 13.6–13.9 of 18 speed/frequency contexts and 5.89–5.93
of six speeds. The local information shortage is mainly **few gain/context
experiments**, rather than permanent absence of an entire speed range.
A claim about route-specific non-IID exposure, unseen vehicles or unseen
trajectory shapes would require different evidence.

In this homogeneous setting the posterior strongly favors fully shared cost
information. The remaining failures are therefore not explained simply by
vehicle heterogeneity. The model shares eligible cost information; the
eligibility estimator still uses only the target's own observations.
Failed-test evidence is not federated through an eligibility model.

A chronological diagnostic finds 16 of D0.5's 70 failed BO requests after
an earlier-round peer failure at the same exact gain/context. All sixteen
fail again. That is a concrete information-use gap, not proof that a new
policy would save sixteen tests: replacement actions and resulting histories
are untested. It motivates examining what the fleet should learn from both
successful and failed controller trials, rather than another bonus sweep.

The older two-speed confirmation also limits the heterogeneous claim.
D/E's homogeneous success benefit passes the declared criterion, whereas
heterogeneous added sharing does not: its success interval includes zero,
paired cost is higher and complete eligibility is lower. A practical D/A
win cannot substitute for the matched D/E control. The new homogeneous
six-speed results do not close that heterogeneous gap.

## The scientific story we can currently defend

A fleet can learn a reusable map from controller gains and operating context
to closed-loop cost **while generating the data online**, under a strict
local experiment budget. Sharing this learned map improves the probability
of finding a near-reference controller beyond what the same peer-gain
catalogue mechanism achieves with independent cost models. Exploration
influences which useful gains are discovered and then transferred, and
also exposes vehicles to additional failed trials.

That is a defensible limited story about online fleet information for
controller calibration. The present evidence does not establish reliable
complete commissioning, general heterogeneous improvement, certified safety,
continuous-domain global BO convergence, formal privacy or a communication
advantage. The family of shared/local models and the model-sharing channel
should remain distinct from the acquisition coefficient selected for a
particular operational preference.

## The bigger-picture decision before more implementation

The exploration ablation is closed. Keep the physics, representation, peer
rule and beta = 0 / 0.5 results frozen while selecting the next question.

There is an information-architecture question before a communication or
privacy claim. The learner's training datum is already a compact scalar J
with a gain/context coordinate. Keeping trajectory logs local does not by
itself justify exchanging dense model statistics instead of compact trial
summaries. The existing peer channel even publishes incumbent gains together
with their scalar costs. A concrete policy on what may be shared, and an
observation-sharing reference where allowed, would clarify what the federated
implementation contributes beyond pooling additional information. The present
evidence establishes the latter's value; it does not establish this message
format as necessary, more private, or communication optimal.

| Direction | Why it is justified | What it would need to establish |
|---|---|---|
| Confirm the current limited claim | D0.5 leads the declared primary metric and has a matched sharing gain | Fresh, fixed-protocol replication of that chosen objective with failures, eligibility and costs retained |
| Learn feasibility information across the fleet | Failed trials are costly and their labels currently remain local | Better success/failure/coverage tradeoffs under matched information rules, rather than just a higher hit rate |
| Reduce communication while holding inference fixed | Reference model payload is about 660 MB per fleet, before transport overhead | A justified information policy, comparison with compact observation exchange where allowed, and lower payload with verified posterior/decision fidelity |
| Revisit physical heterogeneity | Earlier heterogeneous D/E evidence remains conditional | Added value over the matched local model on independent fleets, with complete eligibility retained |

For a stronger claim about **reliable controller commissioning**, I would
prioritize the use of feasibility information before communication or more
physical/task variants. For the narrower claim about near-reference hit rate,
D0.5 is a reasonable frozen candidate for confirmation, followed by
communication work. The choice should follow the intended control objective,
not whichever metric makes one method look best.

The next discussion should select that scientific claim, its operational
success criterion and the permitted information exchange. This consolidation
does not implement another controller
policy, change the cost function or launch another experiment.

## Reproducibility and evidence

The [combined audit](analysis/six_speed_consolidation/verification.json)
replays all **10,368 paid decisions**, reconstructs **36 saved numerical
tables**, preserves every source-result fingerprint and performs no new
controller simulation or hyperparameter optimization. Original A/B fit
states are restored for cross-platform audit; identical fresh refitting is
not claimed. All current six-speed references are available and all online
integration attempts are resolved.

Reproduce the consolidation with:

```bat
python code/scripts/consolidate_tracking_six_speed.py --workers 3
```

Detailed sources:

- [Three-study six-speed audit and latest result](analysis/six_speed_consolidation/report.md).
- [Initial six-speed pilot](analysis/six_speed_contextual_pilot/report.md).
- [Zero-bonus comparison](analysis/six_speed_zero_bonus/report.md).
- [Small-bonus protocol](tracking_six_speed_small_bonus.md).
- [Earlier two-speed confirmation](analysis/spectral_confirmation/report.md).
- [GP-fit replay closure](analysis/gp_replay_closure/report.md).
- [Earlier experiment-chain consolidation](fedbo_consolidation_2026_11_24.md).
