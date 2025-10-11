"""
Setup script for AI-CCTV Waterlogging Detection & Forecasting System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="waterlogging-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered waterlogging detection and forecasting from CCTV footage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/waterlogging-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "opencv-python>=4.8.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "Pillow>=10.0.0",
        "albumentations>=1.3.1",
        "tqdm>=4.65.0",
        "scipy>=1.11.1",
        "segmentation-models-pytorch>=0.3.3",
        "timm>=0.9.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "waterlogging-demo=demo:main",
            "waterlogging-train-detection=train_detection:main",
            "waterlogging-train-forecasting=train_forecasting:main",
        ],
    },
)
