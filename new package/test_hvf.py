"""
Test script for hvstrip-progressive package.
"""
import sys
from pathlib import Path

# Add package to path
pkg_dir = Path(__file__).parent
sys.path.insert(0, str(pkg_dir))

print("=" * 60)
print("Testing HVstrip-Progressive Package")
print("=" * 60)

# Test 1: Check if we can import the modules
print("\n1. Testing imports...")
try:
    from hvstrip_progressive.core import hv_forward
    print("   ✓ Successfully imported hv_forward module")
except Exception as e:
    print(f"   ✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Check if the executable path is correctly detected
print("\n2. Checking HVf executable path...")
try:
    exe_path = hv_forward.DEFAULT_CONFIG["exe_path"]
    print(f"   Detected executable: {exe_path}")
    if Path(exe_path).exists():
        print(f"   ✓ Executable exists and is accessible")
    else:
        print(f"   ✗ Executable not found at path")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Run HVf on example soil profile
print("\n3. Testing HVf computation...")
try:
    example_dir = pkg_dir / "hvstrip_progressive" / "Example"
    model_file = example_dir / "Soil_Profile_Model.txt"

    if not model_file.exists():
        print(f"   ✗ Example model file not found: {model_file}")
        sys.exit(1)

    print(f"   Using model: {model_file}")
    print("   Computing H/V curve...")

    freqs, amps = hv_forward.compute_hv_curve(str(model_file))

    print(f"   ✓ Successfully computed H/V curve")
    print(f"   Number of frequency points: {len(freqs)}")
    print(f"   Frequency range: {min(freqs):.2f} - {max(freqs):.2f} Hz")
    print(f"   Amplitude range: {min(amps):.3f} - {max(amps):.3f}")

    # Find peak
    peak_idx = amps.index(max(amps))
    peak_freq = freqs[peak_idx]
    peak_amp = amps[peak_idx]
    print(f"   Peak H/V: {peak_amp:.3f} at {peak_freq:.2f} Hz")

except Exception as e:
    print(f"   ✗ Error during computation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test layer stripping
print("\n4. Testing layer stripping...")
try:
    from hvstrip_progressive.core import stripper
    import tempfile

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write peeled models to temporary directory
        strip_dir = stripper.write_peel_sequence(str(model_file), tmpdir)
        print(f"   ✓ Created peeled models in: {strip_dir}")

        # Count the created subdirectories
        step_dirs = [d for d in strip_dir.iterdir() if d.is_dir()]
        print(f"   ✓ Generated {len(step_dirs)} peeling steps")

        for step_dir in sorted(step_dirs):
            model_files = list(step_dir.glob("*.txt"))
            if model_files:
                print(f"      {step_dir.name}")

except Exception as e:
    print(f"   ✗ Error during layer stripping: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed successfully!")
print("=" * 60)
print("\nPackage is ready for use!")
