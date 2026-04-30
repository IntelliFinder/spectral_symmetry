# Matched-compute follow-up (#1): map + oap at k=16 / 500 ep / no early stop

## Why

The Sweep 4 saturation test (`results/test_time_averaging_augmented_k16_long/`)
trained \texttt{random\_augmented} at k=16 for 500 epochs with disabled
early stopping and beat the prior 200-epoch \texttt{oap} numbers from
Sweep 3. The adversarial review (2026-04-30) flagged this as
apples-to-oranges: only one side got the extra compute. The headline
"random augmentation beats canonicalization with sufficient compute"
implicitly assumes 500-epoch \texttt{oap} would not also gain — never
tested.

This sweep tests it directly. Same protocol as Sweep 4
(500 ep / patience 999 / no early stop), but on \texttt{map} and
\texttt{oap} instead of \texttt{random\_augmented}.

## Configuration

| Knob | Value |
|---|---|
| Canonicalizations | `map`, `oap` (no `random_augmented` — Sweep 4 already covers it) |
| k | 16 |
| cache-n-eigs | 16 |
| Hidden dim | 16, 128, 512 |
| Seeds | 0, 1, 2 |
| Epochs | 500 |
| Patience | 999 (effectively disabled) |
| Eigval scale | off |
| Test-aug-samples | 16 (no-op for map/oap, harmless) |
| Base-dir | `results/matched_compute_k16/` |

Total: 2 × 3 × 3 = **18 runs**, ~30–40 h wall-clock at 12 GPU slots.

## Decision after this sweep

Compare each cell's matched-compute test AP to Sweep 4's
\texttt{random\_augmented} at the same h:

| h   | rand_aug 500ep | matched-compute oap 500ep | matched-compute map 500ep |
|----:|---------------:|--------------------------:|--------------------------:|
|  16 |        0.0691  |                       TBD |                       TBD |
| 128 |        0.1916  |                       TBD |                       TBD |
| 512 |        0.2483  |                       TBD |                       TBD |

If \texttt{oap}/\texttt{map} 500-ep numbers are below 0.0691 / 0.1916 /
0.2483, the matched-compute conclusion stands: random_augmented is
strictly better at k=16 even when both sides get the same training
budget. If \texttt{oap}/\texttt{map} catch up or surpass, the headline
needs revising.

A separate Launch B follows for k=3 / h=512 (the lone surviving
canonicalization edge from Sweep 1) under the same matched protocol.
