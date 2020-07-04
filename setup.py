from setuptools import find_packages, setup
import os
import re


with open("README.md") as f:
    readme = f.read()


required_packages = ["cv2", "cmake", "dlib"]

setup(
    name="eye-pass",
    version="0.0.1",
    description="Library for load eye-pass",
    long_description=readme,
    author="kunihik0",
    author_email="",
    url="https://github.com/kunihik0/eye-pass",
    install_requires=required_packages,
    packages=find_packages(exclude=[]),
)
