#!/usr/bin/env python
"""
Setup script for hvstrip-progressive package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().splitlines()

setup(
    name="hvstrip-progressive",
    version="1.0.0",
    author="Mersad Fathizadeh",
    author_email="mersadf@uark.edu",
    description="Progressive Layer Stripping Analysis for HVSR (Horizontal-to-Vertical Spectral Ratio)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mersadfathizadeh1995/hvstrip-progressive",
    packages=find_packages(),
    license="GPL-3.0-only",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
        ],
    },
    entry_points={
        "console_scripts": [
            "hvstrip-progressive=hvstrip_progressive.cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "hvstrip_progressive": [
            "config/*.yaml",
            "config/*.yml",
        ],
    },
    keywords="seismology geophysics hvsr layer-stripping diffuse-field",
    project_urls={
        "Bug Reports": "https://github.com/mersadfathizadeh1995/hvstrip-progressive/issues",
        "Source": "https://github.com/mersadfathizadeh1995/hvstrip-progressive",
        "Documentation": "",
    },
)
