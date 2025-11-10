"""
HV computation for HVSR inversion workflow.

Generates H/V curves for specific layer counts, producing files compatible with
the inversion pipeline (combine_hv_profiles_adaptive.py).

Output naming: range_[X-Y]__hv_L{k:02d}.txt for layer-specific processing.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from collections import defaultdict
from multiprocessing import Pool, cpu_count, current_process
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

# ────────────────────────── USER SETTINGS ──────────────────────────
# Base paths for L06 data generation - Using relative paths for portability
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # HVSR_Inversion_L06/
DATA_DIR = BASE_DIR / "Data" / "L06"

# HVf executable for Linux - try multiple candidates
# Priority: HVf_Serial (known working) > HVf > HVf.exe
HVF_CANDIDATES = [
    BASE_DIR / "Data_Gen_Codes" / "HVF_executables" / "HVf_Serial",
    BASE_DIR / "Data_Gen_Codes" / "HVF_executables" / "HVf",
    BASE_DIR / "Data_Gen_Codes" / "HVF_executables" / "HVf.exe",
]
HVF_EXE_PATH = None  # Will be set by find_hvf_executable()
SOIL_PROFILES_DIR = str(DATA_DIR / "soil_profiles")
FREQUENCY_LISTS_DIR = str(DATA_DIR / "frequency_ranges" / "frequency_lists")  # Not used in Linux version
OUTPUT_DIR = str(DATA_DIR / "hv_curves")

# Soil profile file configuration
EXPECTED_SOIL_FILES = None     # Set to None to process all files found
                               # Or set to specific number (e.g., 6 for L06)

# Frequency range configuration - ONLY [0.2-30] for L06
LOWERS = [0.2]
UPPERS = [30.0]

# HVf parameters (Linux version uses different args than Windows .exe)
# Using fmin/fmax/nf instead of frequency file for compatibility
FMIN = "0.2"   # Minimum frequency (Hz)
FMAX = "30.0"  # Maximum frequency (Hz)
NF = "100"     # Number of frequency points
NMR = "1"      # Number of Monte Carlo runs
NML = "1"      # Number of layer models
NKS = "10"     # Number of kappa samples

# Processing parameters
TARGET_LAYER_COUNT = 6        # L06: 6 layers
PROCESS_BATCH_SIZE = 500       # models per parallel batch
TIMEOUT_SECONDS = 30           # timeout per HVf.exe call
ONE_HV_FILE_PER_SOIL_FILE = True  # Create separate HV file for each soil profile file
# ───────────────────────────────────────────────────────────────────


def find_hvf_executable(candidates: List[Path]) -> Path:
    """
    Find a working HVf executable from a list of candidates.
    Tests each one to ensure it's the correct architecture.
    
    Returns the first working executable found.
    """
    import platform
    
    for candidate in candidates:
        if not candidate.exists():
            continue
        
        print(f"[INFO] Checking HVf candidate: {candidate.name}")
        
        # Try to make it executable
        try:
            if not os.access(candidate, os.X_OK):
                current_mode = candidate.stat().st_mode
                candidate.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except:
            pass
        
        # Test if it can execute (quick version check)
        try:
            result = subprocess.run(
                [str(candidate), "-v"],
                capture_output=True,
                text=True,
                timeout=2
            )
            # If it runs without "Exec format error", it's good!
            print(f"[OK] Found working HVf: {candidate.name}")
            return candidate
        except subprocess.TimeoutExpired:
            # Timeout is fine - means it's trying to run (might not support -v flag)
            print(f"[OK] Found working HVf: {candidate.name} (timeout on -v, but executable)")
            return candidate
        except OSError as e:
            if "Exec format error" in str(e):
                print(f"[SKIP] {candidate.name} - Wrong architecture (Exec format error)")
                continue
            else:
                print(f"[WARN] {candidate.name} - Error: {e}")
                continue
    
    raise FileNotFoundError(
        f"No working HVf executable found! Tried: {[c.name for c in candidates]}\n"
        f"Please ensure you have a Linux-compatible HVf binary compiled for your system."
    )


def ensure_executable(path: Path) -> Path:
    """
    Make sure the HVf binary exists and is executable.
    Automatically adds execute permissions if needed.
    
    Based on working implementation from HV-in-not-parallel.py
    """
    if path is None:
        raise FileNotFoundError(
            "No HVf executable found. Check HVF_EXE_PATH in USER SETTINGS."
        )
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HVf executable not found at {path}")
    
    if path.is_dir():
        raise IsADirectoryError(f"Expected an executable file, got a directory: {path}")
    
    # Check if already executable
    if os.access(path, os.X_OK):
        return path
    
    # Try to add execute permissions
    print(f"[INFO] HVf is not executable. Adding execute permissions...")
    try:
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"[OK] Added execute permissions to {path}")
    except PermissionError as exc:
        raise PermissionError(
            f"{path} exists but cannot be made executable. "
            f"Please run manually: chmod +x {path}"
        ) from exc
    
    # Verify it's now executable
    if not os.access(path, os.X_OK):
        raise PermissionError(
            f"{path} is still not executable after chmod. "
            f"Please run manually: chmod +x {path}"
        )
    
    return path


def parse_multiple_models(filepath: Path) -> List[List[str]]:
    """Parse soil profile file into model blocks."""
    with filepath.open('r', encoding='utf-8') as f:
        raw_lines = [ln.rstrip() for ln in f]

    all_models = []
    current_block = []
    for line in raw_lines:
        if not line.strip():
            if current_block:
                all_models.append(current_block)
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        all_models.append(current_block)
    return all_models


def filter_models_by_layer_count(models: List[List[str]], target_layers: int) -> List[List[str]]:
    """Filter model blocks to only include those with target layer count."""
    filtered = []
    for block in models:
        if block:
            try:
                n_including_halfspace = int(block[0])
                n_layers = n_including_halfspace - 1
                if n_layers == target_layers:
                    filtered.append(block)
            except (ValueError, IndexError):
                continue
    return filtered


def parse_hv_data(raw_text: str) -> List[Tuple[float, float]]:
    """Parse HV output into (frequency, hv_amplitude) pairs."""
    lines = raw_text.splitlines()
    tokens = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.extend(line.split())

    if len(tokens) % 2 != 0:
        # Try to handle odd tokens by dropping the last incomplete pair
        if len(tokens) > 0:
            tokens = tokens[:len(tokens) - 1]
        if len(tokens) % 2 != 0:
            print(f"[WARN] Cannot parse HV data - odd tokens after adjustment: {len(tokens)}")
            return []

    pairs = []
    for i in range(0, len(tokens) - 1, 2):
        try:
            freq = float(tokens[i])
            hv = float(tokens[i+1])
            pairs.append((freq, hv))
        except (ValueError, IndexError) as e:
            print(f"[WARN] Failed to parse tokens at index {i}: {e}")
            continue
    
    return pairs


def process_model(task: Tuple[str, int, List[str], str, str]) -> Tuple[str, int, List[Tuple[float, float]]]:
    """Worker function to run HVf for one model (Linux version).
    
    Note: Removed unused freq_list_path parameter - Linux HVf uses fmin/fmax/nf instead.
    """
    prof_name, idx, lines, tmp, hvf_exe_path = task
    pid = current_process().pid
    m_fp = Path(tmp) / f"{prof_name}_model{idx}_{pid}.txt"

    # Write model
    m_fp.write_text("\n".join(lines) + "\n")

    # Run HVf with fmin/fmax/nf parameters (Linux version)
    cmd = [
        hvf_exe_path, "-hv",
        "-f", str(m_fp),
        "-fmin", FMIN,
        "-fmax", FMAX,
        "-nf", NF,
        "-nmr", NMR,
        "-nml", NML,
        "-nks", NKS
    ]
    
    try:
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"[ERR] Timeout for profile={prof_name} model={idx}")
        return (prof_name, idx, [])
    except Exception as e:
        print(f"[ERR] Exception running HVf for profile={prof_name} model={idx}: {e}")
        return (prof_name, idx, [])

    if proc.returncode != 0:
        print(f"[ERR] HVf failed (code {proc.returncode}) profile={prof_name} model={idx}")
        if proc.stderr:
            print(f"      stderr: {proc.stderr[:200]}")
        hv_pairs = []
    else:
        # Linux HVf writes to HV.dat file
        hv_dat = Path(tmp) / "HV.dat"
        if hv_dat.exists():
            raw = hv_dat.read_text()
        else:
            # Fallback to stdout if HV.dat doesn't exist
            raw = proc.stdout
        
        if not raw.strip():
            print(f"[WARN] Empty HV output for profile={prof_name} model={idx}")
            hv_pairs = []
        else:
            hv_pairs = parse_hv_data(raw)
            
            if not hv_pairs:
                print(f"[WARN] No HV pairs parsed for profile={prof_name} model={idx}")
                # Debug: print first 200 chars of raw output
                print(f"      Raw output preview: {raw[:200]}")

    # Clean up
    for path in (m_fp, Path(tmp) / "HV.dat"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    return (prof_name, idx, hv_pairs)


def process_single_soil_file(soil_file: Path, range_tag: str) -> bool:
    """Process one soil profile file and create its corresponding HV output file (Linux version)."""
    layer_tag = f"L{TARGET_LAYER_COUNT:02d}"
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output naming: range_[0.5-20]__hv_L05__Soil_Profile_1.txt
    output_file = output_dir / f"range_{range_tag}__hv_{layer_tag}__{soil_file.stem}.txt"
    
    # Resume check
    if output_file.exists():
        print(f"  [SKIP] {output_file.name} already exists")
        return True
    
    print(f"  [INFO] Processing {soil_file.name}")
    
    # Parse and filter models
    all_models = parse_multiple_models(soil_file)
    filtered_models = filter_models_by_layer_count(all_models, TARGET_LAYER_COUNT)
    
    if not filtered_models:
        print(f"  [WARN] No {TARGET_LAYER_COUNT}-layer models found in {soil_file.name}")
        return False
    
    print(f"  [INFO] Found {len(filtered_models)} models with {TARGET_LAYER_COUNT} layers")
    
    # Create tasks (Linux version doesn't need frequency file)
    tasks = []
    for i, m in enumerate(filtered_models, start=1):
        tasks.append((soil_file.stem, i, m, None, HVF_EXE_PATH))
    
    # Process in batches with progress tracking
    all_results = []
    shared_tmp = tempfile.mkdtemp(prefix=f"hvf_{soil_file.stem}_")
    
    with tqdm(total=len(tasks), desc=f"  {soil_file.name}", 
             unit="model", smoothing=0.1, mininterval=0.5, leave=False) as pbar:
        
        for batch_start in range(0, len(tasks), PROCESS_BATCH_SIZE):
            batch_end = min(batch_start + PROCESS_BATCH_SIZE, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]
            
            # Add shared temp dir
            batch_tasks = [(p, i, m, shared_tmp, exe) for (p, i, m, _, exe) in batch_tasks]
            
            # Optimize worker count
            n_workers = min(cpu_count() * 2, len(batch_tasks), 16)
            
            try:
                with Pool(processes=n_workers) as pool:
                    batch_results = pool.map(process_model, batch_tasks)
                    all_results.extend(batch_results)
                    pbar.update(len(batch_tasks))
            except Exception as e:
                print(f"  [ERROR] Batch {batch_start}-{batch_end} failed: {e}")
                pbar.update(len(batch_tasks))
    
    # Write output file
    with output_file.open("w") as f:
        f.write(f"# H/V curves for frequency range {range_tag}, {TARGET_LAYER_COUNT}-layer models\n")
        f.write(f"# Source: {soil_file.name}\n")
        f.write(f"# Generated by HVf.exe for inversion training ({layer_tag})\n")
        
        for prof, idx, pairs in all_results:
            if pairs:  # Only write if we have data
                f.write(f"\n# --- {prof} model #{idx} ---\n")
                f.write("#Freq[Hz]  H/V\n")
                for frq, hv in pairs:
                    f.write(f"{frq:.6f} {hv:.6f}\n")
    
    print(f"  [OK] Wrote {output_file.name} ({len(all_results)} models)")
    
    # Clean up
    shutil.rmtree(shared_tmp, ignore_errors=True)
    return True


def process_frequency_range(range_tag: str) -> bool:
    """Process one frequency range for the target layer count.
    
    Note: Linux HVf uses fmin/fmax/nf parameters instead of frequency files.
    The frequency file check has been removed as it's not needed for Linux version.
    """
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layer_tag = f"L{TARGET_LAYER_COUNT:02d}"
    soil_dir = Path(SOIL_PROFILES_DIR)
    
    # Get soil profile files. Accept both numeric shards (Soil_Profile_1.txt, ...) and
    # single per-class files (e.g., Soil_Profile_Layering_6.txt) produced by Step 1.
    soil_files = sorted(soil_dir.glob("Soil_Profile_*.txt"))
    
    if not soil_files:
        print(f"[ERROR] No Soil_Profile_*.txt files found in {soil_dir}")
        return False
    
    print(f"\n[INFO] Processing {range_tag} for {TARGET_LAYER_COUNT}-layer models")
    print(f"[INFO] Found {len(soil_files)} soil profile files")
    
    # Validate expected file count
    if EXPECTED_SOIL_FILES is not None:
        if len(soil_files) != EXPECTED_SOIL_FILES:
            print(f"[WARNING] Expected {EXPECTED_SOIL_FILES} files, but found {len(soil_files)}")
            print(f"[WARNING] Files found: {[f.name for f in soil_files]}")
            response = input(f"Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print(f"[ABORT] User cancelled processing")
                return False
        else:
            print(f"[OK] File count matches expected: {EXPECTED_SOIL_FILES} files")
    
    # Process each soil file separately or combined
    if ONE_HV_FILE_PER_SOIL_FILE:
        print(f"[INFO] Mode: Creating separate HV file for each soil profile file")
        success_count = 0
        for soil_file in soil_files:
            if process_single_soil_file(soil_file, range_tag):
                success_count += 1
        
        print(f"\n[SUMMARY] Successfully processed {success_count}/{len(soil_files)} soil files")
        return success_count == len(soil_files)
    
    else:
        # OLD BEHAVIOR: Combine all into single file
        print(f"[INFO] Mode: Creating single combined HV file")
        output_file = output_dir / f"range_{range_tag}__hv_{layer_tag}.txt"
        
        if output_file.exists():
            print(f"[RESUME] Skipping range {range_tag} {layer_tag} (output already exists)")
            return True
        
        # Collect tasks from all soil profile files
        tasks = []
        total_models_found = 0
        
        for pf in soil_files:
            all_models = parse_multiple_models(pf)
            filtered_models = filter_models_by_layer_count(all_models, TARGET_LAYER_COUNT)
            
            for i, m in enumerate(filtered_models, start=1):
                tasks.append((pf.stem, total_models_found + i, m, None, HVF_EXE_PATH))
            
            total_models_found += len(filtered_models)
        
        if not tasks:
            print(f"[WARN] No {TARGET_LAYER_COUNT}-layer models found for range {range_tag}")
            return False
        
        print(f"[INFO] Found {len(tasks)} models with {TARGET_LAYER_COUNT} layers")
        
        # Process in batches
        all_results = []
        shared_tmp = tempfile.mkdtemp(prefix=f"hvf_inversion_{range_tag.replace('[', '').replace(']', '').replace('-', '_')}_")
        
        with tqdm(total=len(tasks), desc=f"Processing {range_tag} L{TARGET_LAYER_COUNT:02d}", 
                 unit="model", smoothing=0.1, mininterval=0.5) as pbar:
            
            for batch_start in range(0, len(tasks), PROCESS_BATCH_SIZE):
                batch_end = min(batch_start + PROCESS_BATCH_SIZE, len(tasks))
                batch_tasks = tasks[batch_start:batch_end]
                
                batch_tasks = [(p, i, m, shared_tmp, exe) for (p, i, m, _, exe) in batch_tasks]
                n_workers = min(cpu_count() * 2, len(batch_tasks), 16)
                
                try:
                    with Pool(processes=n_workers) as pool:
                        batch_results = pool.map(process_model, batch_tasks)
                        all_results.extend(batch_results)
                        pbar.update(len(batch_tasks))
                except Exception as e:
                    print(f"[ERROR] Batch {batch_start}-{batch_end} failed: {e}")
                    pbar.update(len(batch_tasks))
        
        # Organize by profile & write output
        buckets = defaultdict(list)
        for prof, idx, pairs in all_results:
            buckets[prof].append((idx, pairs))
        
        # Write combined HV file
        with output_file.open("w") as f:
            f.write(f"# H/V curves for frequency range {range_tag}, {TARGET_LAYER_COUNT}-layer models\n")
            f.write(f"# Generated by HVf.exe for inversion training ({layer_tag})\n")
            
            for prof, lst in sorted(buckets.items()):
                lst.sort(key=lambda t: t[0])
                for idx, pairs in lst:
                    f.write(f"\n# --- {prof} model #{idx} ---\n")
                    f.write("#Freq[Hz]  H/V\n")
                    for frq, hv in pairs:
                        f.write(f"{frq:.6f} {hv:.6f}\n")
        
        print(f"[OK] {layer_tag} {range_tag}: wrote {output_file}")
        shutil.rmtree(shared_tmp, ignore_errors=True)
        return True


def find_soil_profiles_dir() -> Path:
    """Find the actual soil profiles directory (handle random generation)."""
    base_dir = Path(SOIL_PROFILES_DIR)
    
    # Check if base directory has files directly
    if list(base_dir.glob("Soil_Profile_*.txt")):
        return base_dir
    
    # Look for random_* subdirectories and use the most recent one
    random_dirs = list(base_dir.glob("random_*"))
    if random_dirs:
        latest_dir = max(random_dirs, key=lambda p: p.stat().st_mtime)
        print(f"[INFO] Using most recent random directory: {latest_dir.name}")
        return latest_dir
    
    return base_dir


def main() -> int:
    global HVF_EXE_PATH
    
    print(f"[INFO] Starting HV computation for {TARGET_LAYER_COUNT}-layer inversion")
    
    # Find a working HVf executable
    try:
        hvf_path = find_hvf_executable(HVF_CANDIDATES)
        HVF_EXE_PATH = str(hvf_path)  # Update global with working executable
        print(f"[OK] Using HVf executable: {hvf_path}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    # Find actual soil profiles directory
    actual_soil_dir = find_soil_profiles_dir()
    soil_files = list(actual_soil_dir.glob("Soil_Profile_*.txt"))
    if not soil_files:
        print(f"[ERROR] No Soil_Profile_*.txt files found in {actual_soil_dir}")
        return 1
    
    print(f"[INFO] Using soil profiles from: {actual_soil_dir}")
    
    # Update the global variable for process_frequency_range function
    global SOIL_PROFILES_DIR
    SOIL_PROFILES_DIR = str(actual_soil_dir)

    # Generate frequency range tags from USER SETTINGS
    def _fmt(x: float) -> str:
        return ("%g" % x)
    
    range_tags = []
    for lower in LOWERS:
        for upper in UPPERS:
            if upper > lower:
                range_tags.append(f"[{_fmt(lower)}-{_fmt(upper)}]")
    
    if not range_tags:
        print(f"[ERROR] No valid frequency ranges. Check LOWERS and UPPERS in USER SETTINGS.")
        print(f"        LOWERS = {LOWERS}")
        print(f"        UPPERS = {UPPERS}")
        return 1

    # Find available frequency lists
    freq_dir = Path(FREQUENCY_LISTS_DIR)
    available_ranges = []
    for tag in range_tags:
        freq_file = freq_dir / f"freq_range_{tag}.txt"
        if freq_file.exists():
            available_ranges.append(tag)
        else:
            print(f"[INFO] Frequency file not found (skipping): {freq_file.name}")

    if not available_ranges:
        print(f"[ERROR] No frequency range files found in {freq_dir}")
        return 1

    print(f"[INFO] Found {len(soil_files)} soil files, {len(available_ranges)} available frequency ranges")
    print(f"[INFO] Target layer count: {TARGET_LAYER_COUNT}")
    print(f"[INFO] Will process ranges: {available_ranges}")

    # Process each available frequency range
    completed_ranges = 0
    layer_tag = f"L{TARGET_LAYER_COUNT:02d}"
    
    with tqdm(available_ranges, desc=f"Frequency Ranges ({layer_tag})", unit="range") as range_pbar:
        for range_tag in range_pbar:
            range_pbar.set_description(f"Processing {range_tag} ({layer_tag})")
            success = process_frequency_range(range_tag)
            if success:
                completed_ranges += 1
                range_pbar.set_postfix(completed=f"{completed_ranges}/{len(available_ranges)}")
            else:
                range_pbar.set_postfix(completed=f"{completed_ranges}/{len(available_ranges)}", status="FAILED")

    print(f"\n[SUMMARY] Completed {completed_ranges}/{len(available_ranges)} frequency ranges for {layer_tag}")
    print(f"HV curve files saved in: {Path(OUTPUT_DIR).resolve()}")
    
    return 0 if completed_ranges == len(available_ranges) else 1


if __name__ == "__main__":
    raise SystemExit(main())
