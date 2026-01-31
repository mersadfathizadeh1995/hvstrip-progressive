"""
Make_Profiles.py
================
Extract Vs, Vp and/or density profiles from a Geopsy *.report file.

Model-selection modes
---------------------
SELECTION_MODE = "best"
    gpdcreport -best N_BEST_MODELS  <report>
SELECTION_MODE = "misfit"
    gpdcreport -m MISFIT_MAX -n N_MAX_MODELS  <report>

The resulting models are piped to gpprofile to generate plain-text profile
tables ready for later plotting or statistics.
"""

import os
import subprocess
from pathlib import Path

# ───────────────────────── USER SETTINGS ────────────────────────────
SITE_NAME   = "Indep_Powerplant_B6"     # appears in output filenames

REPORT_FILE = r"D:\Runs\Independence_Plant\Phase_1\B_6\Processed\Dinver\B6_Dinver\B6_Final\Dinver_Inde_B6_reports\run_23.report"
# OUTPUT_DIR  = r"C:\Users\mersadf\Desktop\Reports\6-layer\Ground_profile"
OUTPUT_DIR  = r"D:\Research\Nsf_Project\NSF\Layer_estimator\Osark\Workspace\HV_Truncate\Codes\L05_Truncate\11_MDN_Integrated_Interface\Data_prep\Tests"
GEOPSY_BIN  = r"C:\Geopsy.org\bin"    # gpdcreport.exe etc.
GIT_BASH_EXE = r"C:\Users\mersadf\AppData\Local\Programs\Git\git-bash.exe"

# Which profile types to generate  (any subset of vs / vp / rho)
# PROFILES_TO_MAKE = ["vs", "vp", "rho"]
PROFILES_TO_MAKE = ["vs", "vp", "rho"]
# ---- model-selection mode ------------------------------------------
SELECTION_MODE = "best"           # "best"  or  "misfit"

# • best-N settings
N_BEST_MODELS  = 1

# • misfit-cutoff settings    (used only if SELECTION_MODE == "misfit")
MISFIT_MAX     = 1            # keep models with misfit ≤ this
N_MAX_MODELS   = 1000             # stream at most this many

# ---- gpprofile depth grid ------------------------------------------
DEPTH_MAX  = 1000     # m  (None → no -max-depth)
DEPTH_STEP = None     # m  (None → keep raw layering)

# Delete existing *_Vs.txt etc.?    True = overwrite
OVERWRITE  = True
# ────────────────────────────────────────────────────────────────────


# ════════════════  INTERNAL HELPERS  ════════════════════════════════
def to_bash(path: str) -> str:
    """Win path → Git-Bash style (/c/Users/…)."""
    p = Path(path).resolve()
    return f"/{p.drive[0].lower()}{p.as_posix()[2:]}"


def gpdcreport_cmd(report_bash: str) -> str:
    """Return gpdcreport command fragment based on selection mode."""
    if SELECTION_MODE.lower() == "best":
        return f"gpdcreport.exe -best {N_BEST_MODELS} {report_bash}"
    elif SELECTION_MODE.lower() == "misfit":
        return (f"gpdcreport.exe -m {MISFIT_MAX} "
                f"-n {N_MAX_MODELS} {report_bash}")
    else:
        raise ValueError("SELECTION_MODE must be 'best' or 'misfit'")


def gpprofile_cmd(kind: str) -> str:
    flag = {"vs": "-vs", "vp": "-vp", "rho": "-rho"}[kind]
    depth_opts = ""
    if DEPTH_MAX is not None:
        depth_opts += f" -max-depth {DEPTH_MAX}"
    if DEPTH_STEP is not None:
        depth_opts += f" -resample -d {DEPTH_STEP}"
    return f"gpprofile.exe {flag}{depth_opts}"


def run_bash(cmd: str) -> subprocess.CompletedProcess:
    """Run <cmd> through Git Bash; capture stdout/stderr."""
    return subprocess.run([GIT_BASH_EXE, "-c", cmd],
                          capture_output=True, text=True)


# ═════════════════════  MAIN  ═══════════════════════════════════════
def main():
    # sanity checks
    git_ok = Path(GIT_BASH_EXE).is_file()
    if not git_ok:
        raise RuntimeError(f"Git Bash not found: {GIT_BASH_EXE}")
    rpt_path = Path(REPORT_FILE).resolve()
    if not rpt_path.is_file():
        raise RuntimeError(f"Report not found: {REPORT_FILE}")

    out_dir = Path(OUTPUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # PATH for Geopsy tools
    env_prefix = f'export PATH="{to_bash(GEOPSY_BIN)}:$PATH" && '

    rpt_bash   = to_bash(rpt_path)
    select_str = gpdcreport_cmd(rpt_bash)

    wanted = [k.lower() for k in PROFILES_TO_MAKE]
    invalid = [k for k in wanted if k not in ("vs", "vp", "rho")]
    if invalid:
        raise ValueError(f"Unknown profile type(s): {invalid}")

    print(f"\n[INFO] Source report : {REPORT_FILE}")
    print(f"[INFO] Output folder : {out_dir}")
    print(f"[INFO] Mode          : {SELECTION_MODE}\n")

    for k in wanted:
        out_file  = out_dir / f"{SITE_NAME}_{k}.txt"
        if out_file.exists() and not OVERWRITE:
            print(f"[SKIP] {out_file.name} exists (overwrite=False)")
            continue

        bash_cmd = (
            env_prefix +
            f"{select_str} | {gpprofile_cmd(k)} > '{to_bash(out_file)}'"
        )

        print(f"- {out_file.name} …", end=" ", flush=True)
        res = run_bash(bash_cmd)
        if res.returncode == 0:
            print("[OK]")
        else:
            print("[FAIL]")
            print("  stderr:", res.stderr.strip() or "(empty)")

    print("\n[DONE] Profile extraction finished.")


if __name__ == "__main__":
    main()
