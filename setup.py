"""Setup Torch Mamba."""

from setuptools import find_packages, setup


setup(
    name="logfa",
    version="0.1.0",
    description="torch realization of multi-dimensional logarithmic flash attention",
    packages=find_packages(),
    python_requires=">=3.8",
)