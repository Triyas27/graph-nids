# Week 3, Day 5 -- Mean Time to Detection (MTTD)

Not F1, not AUC -- the metric a SOC actually watches: how long from compromise until the model raises an alert, and how much damage is already done by then.

**Compromise start**: 2017-07-06 14:19:00

## Detection timing

| | time | latency from compromise | pivot targets already reached |
|---|---|---|---|
| First alert (any flow) | 2017-07-06 15:00:00 | 0 days 00:41:00 | 1 / 9 |
| First alert ON the pivot chain | 2017-07-06 15:14:00 | 0 days 00:55:00 | 9 / 9 |

**The model does alert earlier than it "should" get credit for** -- its first alert on this host's post-compromise traffic comes at 2017-07-06 15:00:00 (0 days 00:41:00 after compromise), 0 days 00:14:00 before it flags anything actually on the pivot chain. That earlier alert isn't on a real lateral-movement flow (likely a false positive in the large burst of incidental external traffic this host generates), but in a real SOC it would still be the thing that puts an analyst's eyes on this host in the first place -- worth noting as a practical mitigating factor even though it doesn't count as a correct detection in the F1/recall numbers.

**The core result**: by the time the model flags anything actually tied to the lateral-movement chain, **9 of the 9 internal pivot targets have already been reached** -- detection happens at (or after) the compromise is essentially complete, not while it's in progress. This is the practical cost of the dilution problem diagnosed in Week 2: a detector built on hourly aggregates can't intervene mid-chain because it isn't resolving individual flows at all.

## Pivot chain reference

| Destination IP   | first_contact       |
|:-----------------|:--------------------|
| 192.168.10.5     | 2017-07-06 14:33:00 |
| 192.168.10.9     | 2017-07-06 15:07:00 |
| 192.168.10.12    | 2017-07-06 15:08:00 |
| 192.168.10.14    | 2017-07-06 15:09:00 |
| 192.168.10.15    | 2017-07-06 15:10:00 |
| 192.168.10.16    | 2017-07-06 15:10:00 |
| 192.168.10.19    | 2017-07-06 15:12:00 |
| 192.168.10.25    | 2017-07-06 15:13:00 |
| 192.168.10.51    | 2017-07-06 15:14:00 |
