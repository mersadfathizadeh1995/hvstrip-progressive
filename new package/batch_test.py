"""
Batch test script for multiple soil profiles.
Tests the complete workflow: stripping + HV computation.
"""
import sys
from pathlib import Path
import time

# Add package to path
pkg_dir = Path(__file__).parent
sys.path.insert(0, str(pkg_dir))

from hvstrip_progressive.core import hv_forward, stripper


def test_profile(profile_path, output_base):
    """Test a single profile: compute HV and perform layer stripping."""
    profile_name = profile_path.stem
    print(f"\n{'='*60}")
    print(f"Testing Profile: {profile_name}")
    print(f"{'='*60}")

    results = {
        "profile": profile_name,
        "path": str(profile_path),
        "success": False,
        "error": None,
    }

    try:
        # 1. Compute H/V for original model
        print("\n1. Computing H/V curve for original model...")
        start_time = time.time()
        freqs, amps = hv_forward.compute_hv_curve(str(profile_path))
        hv_time = time.time() - start_time

        # Find peak
        peak_idx = amps.index(max(amps))
        peak_freq = freqs[peak_idx]
        peak_amp = amps[peak_idx]

        print(f"   ✓ Computed in {hv_time:.2f}s")
        print(f"   Frequency range: {min(freqs):.2f} - {max(freqs):.2f} Hz")
        print(f"   Peak H/V: {peak_amp:.3f} at {peak_freq:.2f} Hz")

        results["hv_time"] = hv_time
        results["peak_freq"] = peak_freq
        results["peak_amp"] = peak_amp
        results["n_freqs"] = len(freqs)

        # 2. Perform layer stripping
        print("\n2. Performing layer stripping...")
        start_time = time.time()
        profile_output_dir = output_base / profile_name
        strip_dir = stripper.write_peel_sequence(
            str(profile_path),
            str(profile_output_dir)
        )
        strip_time = time.time() - start_time

        # Count steps
        step_dirs = [d for d in strip_dir.iterdir() if d.is_dir()]
        print(f"   ✓ Created {len(step_dirs)} peeling steps in {strip_time:.2f}s")

        results["strip_time"] = strip_time
        results["n_steps"] = len(step_dirs)
        results["strip_dir"] = str(strip_dir)

        # 3. Compute H/V for each stripped model
        print("\n3. Computing H/V curves for stripped models...")
        start_time = time.time()
        step_results = []

        for step_dir in sorted(step_dirs):
            step_name = step_dir.name
            model_files = list(step_dir.glob("*.txt"))

            if not model_files:
                continue

            model_file = model_files[0]
            try:
                step_freqs, step_amps = hv_forward.compute_hv_curve(str(model_file))
                step_peak_idx = step_amps.index(max(step_amps))
                step_peak_freq = step_freqs[step_peak_idx]
                step_peak_amp = step_amps[step_peak_idx]

                step_results.append({
                    "step": step_name,
                    "peak_freq": step_peak_freq,
                    "peak_amp": step_peak_amp,
                })

                print(f"   ✓ {step_name}: Peak={step_peak_amp:.3f} @ {step_peak_freq:.2f} Hz")

            except Exception as e:
                print(f"   ✗ {step_name}: Failed - {e}")
                step_results.append({
                    "step": step_name,
                    "error": str(e),
                })

        batch_hv_time = time.time() - start_time
        print(f"   ✓ Computed {len(step_results)} H/V curves in {batch_hv_time:.2f}s")

        results["batch_hv_time"] = batch_hv_time
        results["step_results"] = step_results
        results["success"] = True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)

    return results


def main():
    print("="*70)
    print("Batch Testing: HVstrip-Progressive Package")
    print("="*70)

    # Setup paths
    profiles_dir = pkg_dir / "hvstrip_progressive" / "Example" / "profiles"
    output_base = pkg_dir / "test_results"
    output_base.mkdir(exist_ok=True)

    if not profiles_dir.exists():
        print(f"\nError: Profiles directory not found: {profiles_dir}")
        print("Please run parse_soil_profiles.py first!")
        sys.exit(1)

    # Get all profile files
    profile_files = sorted(profiles_dir.glob("profile_*.txt"))

    if not profile_files:
        print(f"\nError: No profile files found in {profiles_dir}")
        sys.exit(1)

    print(f"\nFound {len(profile_files)} profiles to test")
    print(f"Output directory: {output_base}")

    # Test each profile
    all_results = []
    successful = 0
    failed = 0

    total_start = time.time()

    for profile_file in profile_files:
        result = test_profile(profile_file, output_base)
        all_results.append(result)

        if result["success"]:
            successful += 1
        else:
            failed += 1

    total_time = time.time() - total_start

    # Summary
    print("\n" + "="*70)
    print("BATCH TEST SUMMARY")
    print("="*70)
    print(f"\nTotal profiles tested: {len(profile_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per profile: {total_time/len(profile_files):.2f}s")

    # Detailed summary
    print("\n" + "-"*70)
    print("Profile Details:")
    print("-"*70)
    print(f"{'Profile':<15} {'Status':<10} {'Layers':<8} {'Peak Freq':<12} {'Peak Amp':<10}")
    print("-"*70)

    for result in all_results:
        profile = result["profile"]
        status = "✓ OK" if result["success"] else "✗ FAIL"
        n_steps = result.get("n_steps", "N/A")
        peak_freq = f"{result.get('peak_freq', 0):.2f}" if result["success"] else "N/A"
        peak_amp = f"{result.get('peak_amp', 0):.3f}" if result["success"] else "N/A"

        print(f"{profile:<15} {status:<10} {n_steps!s:<8} {peak_freq:<12} {peak_amp:<10}")

    print("-"*70)

    if failed == 0:
        print("\n✓ All tests passed successfully!")
    else:
        print(f"\n⚠ {failed} test(s) failed")

    print("="*70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
