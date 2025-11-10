"""
Generate a comprehensive PDF report combining all visualizations and analysis.
"""
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import textwrap

# Add package to path
pkg_dir = Path(__file__).parent
sys.path.insert(0, str(pkg_dir))


def create_title_page(pdf, report_title="HVSR Progressive Layer Stripping Analysis"):
    """Create a professional title page for the PDF report."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.7, report_title,
           ha='center', va='center', fontsize=24, fontweight='bold',
           transform=ax.transAxes)

    # Subtitle
    ax.text(0.5, 0.62, "Complete Analysis Report",
           ha='center', va='center', fontsize=16,
           transform=ax.transAxes, style='italic')

    # Date
    today = datetime.now().strftime("%B %d, %Y")
    ax.text(0.5, 0.5, f"Report Generated: {today}",
           ha='center', va='center', fontsize=12,
           transform=ax.transAxes)

    # Description box
    description = """
    This report presents the results of a progressive layer stripping
    analysis applied to multiple soil profiles using the HVSR (Horizontal-
    to-Vertical Spectral Ratio) method based on diffuse-field theory.

    The analysis systematically removes layers from each profile to
    identify controlling interfaces and understand the contribution of
    different depth ranges to the observed HVSR curve.
    """

    wrapped_text = textwrap.fill(description.strip(), width=70)
    ax.text(0.5, 0.35, wrapped_text,
           ha='center', va='top', fontsize=10,
           transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))

    # Footer
    ax.text(0.5, 0.1, "hvstrip-progressive v1.0.0",
           ha='center', va='center', fontsize=10,
           transform=ax.transAxes, style='italic', color='gray')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_summary_page(pdf, summary_file):
    """Create a summary page with text content."""
    # Read summary file
    with open(summary_file, 'r') as f:
        content = f.read()

    # Split content into sections
    lines = content.split('\n')

    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Use monospace font for better alignment
    ax.text(0.05, 0.95, content,
           ha='left', va='top', fontsize=7,
           transform=ax.transAxes,
           family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.8))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_section_divider(pdf, section_title, section_description=""):
    """Create a section divider page."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Section title
    ax.text(0.5, 0.6, section_title,
           ha='center', va='center', fontsize=20, fontweight='bold',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=1.5', facecolor='steelblue', alpha=0.3))

    # Section description
    if section_description:
        wrapped_text = textwrap.fill(section_description, width=70)
        ax.text(0.5, 0.45, wrapped_text,
               ha='center', va='top', fontsize=11,
               transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def add_image_to_pdf(pdf, image_path, title=""):
    """Add an image to the PDF with optional title."""
    if not Path(image_path).exists():
        print(f"  Warning: Image not found: {image_path}")
        return

    fig = plt.figure(figsize=(8.5, 11))

    if title:
        # Add title at top
        fig.text(0.5, 0.95, title, ha='center', fontsize=14, fontweight='bold')
        # Image takes rest of page
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
    else:
        # Image takes full page
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])

    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis('off')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def generate_comprehensive_pdf_report(output_path):
    """Generate the complete PDF report."""
    print("="*70)
    print("PDF Report Generator")
    print("="*70)

    # Paths
    viz_dir = pkg_dir / "visualization_output"
    plots_dir = viz_dir / "plots"
    summary_file = viz_dir / "analysis_summary.txt"

    if not viz_dir.exists():
        print(f"\nError: Visualization output directory not found: {viz_dir}")
        print("Please run generate_visualizations.py first!")
        return 1

    print(f"\nGenerating PDF report: {output_path}")
    print("This may take a minute...\n")

    with PdfPages(output_path) as pdf:
        # Page 1: Title page
        print("  Adding title page...")
        create_title_page(pdf)

        # Page 2: Table of contents / Summary
        if summary_file.exists():
            print("  Adding summary report...")
            create_summary_page(pdf, summary_file)

        # Section 1: Cross-Profile Comparison
        print("  Adding cross-profile comparison...")
        create_section_divider(
            pdf,
            "Cross-Profile Comparison",
            "Statistical comparison of all analyzed soil profiles, showing peak "
            "frequencies, amplitudes, and layer distributions."
        )

        comparison_plot = plots_dir / "all_profiles_comparison.png"
        if comparison_plot.exists():
            add_image_to_pdf(pdf, comparison_plot)

        # Section 2: Individual Profile Analyses
        print("  Adding individual profile analyses...")
        create_section_divider(
            pdf,
            "Individual Profile Analyses",
            "Detailed analysis for each soil profile showing the complete "
            "progressive layer stripping sequence and H/V curve evolution."
        )

        # Add all individual profile plots
        profile_plots = sorted(plots_dir.glob("profile_*_analysis.png"))
        for i, plot_path in enumerate(profile_plots, 1):
            profile_name = plot_path.stem.replace("_analysis", "").replace("_", " ").title()
            print(f"    Adding {profile_name}...")
            add_image_to_pdf(pdf, plot_path, title=f"{profile_name} Analysis")

        # Metadata
        d = pdf.infodict()
        d['Title'] = 'HVSR Progressive Layer Stripping Analysis Report'
        d['Author'] = 'hvstrip-progressive v1.0.0'
        d['Subject'] = 'Geophysical Analysis Report'
        d['Keywords'] = 'HVSR, Layer Stripping, Geophysics, Soil Profile'
        d['CreationDate'] = datetime.now()

    print(f"\n✓ PDF report generated successfully!")
    print(f"  Output: {output_path}")
    print(f"  Pages: {len(profile_plots) + 4}")  # title + summary + 2 dividers + profiles + comparison

    return 0


def main():
    """Main PDF generation workflow."""
    output_path = pkg_dir / "visualization_output" / "HVSR_Analysis_Report.pdf"

    try:
        result = generate_comprehensive_pdf_report(output_path)

        if result == 0:
            print("\n" + "="*70)
            print("PDF REPORT GENERATION COMPLETE")
            print("="*70)
            print(f"\nFull report available at:")
            print(f"  {output_path}")
            print("\nThis report includes:")
            print("  • Title page")
            print("  • Summary statistics")
            print("  • Cross-profile comparison")
            print("  • Individual profile analyses")
            print("="*70)

        return result

    except Exception as e:
        print(f"\n✗ Error generating PDF report: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
