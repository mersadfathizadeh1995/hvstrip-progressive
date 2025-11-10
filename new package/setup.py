"""
Setup script for hvstrip-progressive package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="hvstrip-progressive",
    version="1.0.0",
    author="HVSR-Diffuse Development Team",
    author_email="your.email@example.com",
    description="HVSR Progressive Layer Stripping Analysis Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hvstrip-progressive",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "matplotlib>=3.5.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "hvstrip=hvstrip_progressive.cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "hvstrip_progressive": [
            "bin/exe_Linux/*",
            "bin/exe_Win/*",
            "Example/*",
        ],
    },
)
