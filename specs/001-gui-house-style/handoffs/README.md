# Handoffs — stage order

The implementer works the tasks.md stages IN ORDER; each stage leaves BOTH suites green
(`pytest tests` legacy-PyQt5 · `pytest tests_v2` PySide6 — separate processes) and the
`tests/golden/` digests bit-identical.

| Stage | Tasks | Gate to the next |
|---|---|---|
| S1 scaffold + AppState | T01–T05 | hygiene/theme/state tests green |
| S2 shell | T06–T08 | renders + bat smoke |
| S3 Forward | T09–T12 | GUI==api==golden parity |
| S4 Strip | T13–T18 | golden parity + steps-as-layers live |
| S5 Research | T19 | concurrency + cancel |
| S6 Hub + cutover | T20–T24 | the full final checklist (tasks T24) |

Copy-sources: `packages/invert_hvsr/invert_hvsr/gui/**` (newest exemplar) + the
`gui-house-style` plugin `references/templates/`. Design law:
`HV_Pro_docs/hvstrip_gui/{SKETCH.txt, DECISIONS.md}`. Never import legacy
`gui/` modules from v2 (DC-4). Frozen compute per `.claude/rules/core.md`.
