#!/usr/bin/env python
"""
Example script demonstrating hvstrip-progressive package usage.
"""

from pathlib import Path
import sys

# Add package to path for development
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from hvstrip_progressive.core import batch_workflow, report_generator


def main():
    """Run example analysis."""
    # Paths
    example_dir = Path(__file__).parent
    model_file = example_dir / "example_model.txt"
    output_dir = example_dir / "output"
    
    print("🎯 HVSR Progressive Layer Stripping - Example")
    print("=" * 60)
    
    # Configuration
    config = {
        "hv_forward": {
            "exe_path": "HVf.exe",  # Update this path as needed
            "fmin": 0.2,
            "fmax": 20.0,
            "nf": 71
        }
    }
    
    try:
        # Step 1: Run complete workflow
        print("\n🚀 Running complete workflow...")
        results = batch_workflow.run_complete_workflow(
            str(model_file),
            str(output_dir),
            config
        )
        
        if not results.get("success", False):
            print("❌ Workflow failed!")
            return 1
        
        # Step 2: Generate comprehensive reports
        print("\n📊 Generating comprehensive reports...")
        strip_dir = output_dir / "strip"
        
        if strip_dir.exists():
            reporter = report_generator.ProgressiveStrippingReporter(
                str(strip_dir),
                str(output_dir / "reports")
            )
            
            report_files = reporter.generate_comprehensive_report()
            
            print(f"\n✅ Example completed successfully!")
            print(f"📁 Results saved in: {output_dir}")
            print(f"📊 Generated {len(report_files)} report files")
        else:
            print("⚠️  Strip directory not found, skipping report generation")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
