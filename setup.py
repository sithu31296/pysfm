import os
import pkg_resources
from setuptools import setup, find_packages


setup(
    name="pysfm",
    py_modules=['pysfm'],
    version="0.1",
    description="Python Only Structure-from-Motion and 3D Computer Vision",
    author="Sithu Aung",
    packages=find_packages(include="pysfm"),
    include_package_data=True
)