"""Setup Torch multi-dimensional logarithmic flash attention """

from setuptools import find_packages, setup


setup(
    name="loga-torch",
    version="0.1.0",
    description="torch realization of multi-dimensional logarithmic flash attention",
    packages=find_packages(),
    python_requires=">=3.8",
)