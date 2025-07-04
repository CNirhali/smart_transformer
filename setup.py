"""
Setup script for Smart Transformer

This script handles the installation and distribution of the Smart Transformer package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version
def get_version():
    version_file = os.path.join("smart_transformer", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="smart-transformer",
    version=get_version(),
    author="Smart Transformer Team",
    author_email="support@smart-transformer.com",
    description="An adaptive and intelligent transformer architecture that outperforms existing transformers",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/smart-transformer",
    project_urls={
        "Bug Reports": "https://github.com/your-username/smart-transformer/issues",
        "Source": "https://github.com/your-username/smart-transformer",
        "Documentation": "https://smart-transformer.readthedocs.io/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "gpu": [
            "flash-attn>=2.3.0",
            "xformers>=0.0.20",
        ],
        "full": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "optuna>=3.2.0",
            "lion-pytorch>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "smart-transformer=smart_transformer.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "smart_transformer": ["*.json", "*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords=[
        "transformer",
        "attention",
        "deep-learning",
        "machine-learning",
        "nlp",
        "natural-language-processing",
        "adaptive",
        "intelligent",
        "pytorch",
    ],
) 