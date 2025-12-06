"""
Setup script for Santander Customer Transaction Prediction project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="santander-customer-transaction-prediction",
    version="1.0.0",
    author="Hasan Kirtas",
    description="Machine learning project for Santander Customer Transaction Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hasankirtas/Santander-Customer-Transaction-Prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "santander-train=scripts.train_pipeline:main",
            "santander-predict=scripts.predict_pipeline:main",
        ],
    },
)

