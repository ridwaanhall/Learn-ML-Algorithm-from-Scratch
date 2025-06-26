"""
Setup script for Learn-ML-Algorithm-from-Scratch package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="learn-ml-from-scratch",
    version="1.0.0",
    author="Ridwan Hall",
    author_email="contact@ridwaanhall.com",
    description="A comprehensive educational package for learning machine learning algorithms from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ridwaanhall/Learn-ML-Algorithm-from-Scratch",
    project_urls={
        "Website": "https://ridwaanhall.com",
        "Bug Tracker": "https://github.com/ridwaanhall/Learn-ML-Algorithm-from-Scratch/issues",
        "Documentation": "https://github.com/ridwaanhall/Learn-ML-Algorithm-from-Scratch/tree/main/docs",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    keywords=[
        "machine learning",
        "education",
        "algorithms",
        "numpy",
        "linear regression",
        "optimization",
        "gradient descent",
        "educational",
        "from scratch",
        "tutorial",
    ],
    package_data={
        "": ["*.md", "*.txt"],
    },
    include_package_data=True,
)
