# Rule: `api/` — the ONE facade (HVStripAnalysis)

Loaded when you touch `HV_Strip_Progressive/api/**`. Qt-free, always.

## Conventions
- **`HVStripAnalysis` is the single session facade** — stateful (config + profiles + results),
  every method returns a JSON-serialisable dict envelope. Extend IT; never add a second facade.
- Config = the `api/config.py` dataclass tree (root `HVStripConfig`). Saved payloads carry
  `config_version: 2`; legacy shapes route through `HVStripConfig.from_legacy_gui_dict()`
  (fixture: `tests/golden/legacy_gui_config.json`). `api/session_io.py` is the ONE load funnel
  for settings.yaml / sessions / project payloads.
- Long ops take `progress_cb: Callable[[dict], None] = None` (default None = byte-identical
  behavior). Streaming comes from the stdout tee over core's narration (`api/_progress.py`) +
  coarse api-level frames; frames are STEP-granular, never per adaptive-rescan call.
- `check_engines()` = existence probes only (no subprocess spawn at import/probe time).
- `research/` consumes this api — keep `run_stripping`/`HVStripConfig` import paths stable or
  update research/ in the same change.
- Every new op ships with a `tests_v2/test_api` test (envelope shape + numbers where applicable).

## Don't
- Import Qt or `gui/`; return non-serialisable objects from public methods; add a parallel
  config representation; or let a facade change drift the golden digests.
