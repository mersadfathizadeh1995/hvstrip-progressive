"""
Parse and separate multiple soil profiles from Soil_Profiles.txt
"""
import sys
from pathlib import Path

# Add package to path
pkg_dir = Path(__file__).parent
sys.path.insert(0, str(pkg_dir))

def parse_multiple_profiles(input_file):
    """
    Parse a file containing multiple soil profiles.

    Each profile starts with a number indicating the number of layers,
    followed by that many lines of layer data, then an empty line.
    """
    profiles = []
    current_profile = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines between profiles
            if not line:
                if current_profile:
                    profiles.append(current_profile)
                    current_profile = []
                continue

            current_profile.append(line)

        # Don't forget the last profile if file doesn't end with empty line
        if current_profile:
            profiles.append(current_profile)

    return profiles


def validate_profile(profile_lines):
    """Validate a single profile's structure."""
    if not profile_lines:
        return False, "Empty profile"

    try:
        n_layers = int(profile_lines[0])
    except ValueError:
        return False, "First line must be number of layers"

    if len(profile_lines) != n_layers + 1:
        return False, f"Expected {n_layers+1} lines (header + {n_layers} layers), got {len(profile_lines)}"

    # Check last layer has thickness 0 (halfspace)
    last_layer = profile_lines[-1].split()
    if len(last_layer) < 4:
        return False, "Last layer must have 4 values"

    if float(last_layer[0]) != 0:
        return False, "Last layer must be halfspace (thickness = 0)"

    return True, "Valid"


def save_profiles(profiles, output_dir):
    """Save each profile as a separate file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for i, profile in enumerate(profiles, 1):
        # Validate profile first
        valid, msg = validate_profile(profile)
        if not valid:
            print(f"   ⚠ Profile {i} is invalid: {msg}")
            continue

        # Save to file
        filename = f"profile_{i:02d}.txt"
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            f.write('\n'.join(profile) + '\n')

        saved_files.append(filepath)

        # Get profile info
        n_layers = int(profile[0])
        print(f"   ✓ Saved Profile {i}: {n_layers} layers → {filename}")

    return saved_files


if __name__ == "__main__":
    print("=" * 60)
    print("Parsing Multiple Soil Profiles")
    print("=" * 60)

    # Input file
    input_file = pkg_dir / "hvstrip_progressive" / "Example" / "Soil_Profiles.txt"

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"\nInput file: {input_file}")

    # Parse profiles
    print("\nParsing profiles...")
    profiles = parse_multiple_profiles(input_file)
    print(f"Found {len(profiles)} profiles")

    # Create output directory
    output_dir = pkg_dir / "hvstrip_progressive" / "Example" / "profiles"

    # Save profiles
    print("\nSaving individual profile files...")
    saved_files = save_profiles(profiles, output_dir)

    print("\n" + "=" * 60)
    print(f"✓ Successfully parsed and saved {len(saved_files)} profiles")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
