"""
Setup configuration for Ontogentelechy
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="ontogentelechy",
    version="0.1.0",
    author="O9NN",
    author_email="dev@o9nn.org",
    description="Purpose-driven cognitive development framework integrating ontogenesis, teleology, and entelechy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/o9nn/ontogentelechy",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    keywords=[
        "cognitive-architecture",
        "teleology",
        "ontogenesis",
        "entelechy",
        "purpose-driven",
        "actualization",
        "developmental-systems",
        "artificial-intelligence",
        "evolutionary-computation",
        "emergence",
    ],
    project_urls={
        "Bug Reports": "https://github.com/o9nn/ontogentelechy/issues",
        "Source": "https://github.com/o9nn/ontogentelechy",
        "Documentation": "https://github.com/o9nn/ontogentelechy/tree/main/docs",
    },
)
