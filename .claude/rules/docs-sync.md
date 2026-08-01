# Rule: docs never drift from code

Loaded every session. Any change to structure, public surface, the data model, or the
architecture MUST update the docs **in the same commit**.

| If you change… | Also update (same commit) |
|---|---|
| The layer layout / where things live | `CLAUDE.md` (the tree) |
| The data contract (Layer/SoilProfile, the HVf format, envelopes, the output tree) | `.context/STRIP_MODEL.md` |
| The architecture — api extensions, gui/v2 progress, consumer rewiring, Hub wiring | `.context/ARCHITECTURE_ASSESSMENT.md` **+** `CLAUDE.md` |
| A layer's boundary / conventions | the matching `.claude/rules/<area>.md` |
| Domain terms | `.context/GLOSSARY.md` |
| The frozen-compute set or the golden fixtures | `CLAUDE.md` + `.claude/rules/{core,tests}.md` + user sign-off |
| The GUI shell / theme (v2) | `HV_Pro_docs/hvstrip_gui/` decisions + `.context/ARCHITECTURE_ASSESSMENT.md` |

This package is MID-UPLIFT — the assessment doc is the plan of record; keep its phase log current.
The repo-level `hvsr_pro/.context/HVSTRIP_PROGRESSIVE_CONTEXT.md` is a POINTER here — never let it
grow content again (it went stale once already and actively misled agents).
