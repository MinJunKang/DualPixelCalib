#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="DPCalib",
    version="0.0.1",
    description="Dual Pixel Calibration Codebase",
    author="Minjun Kang",
    author_email="kmmj2005@kaist.ac.kr",
    url="https://github.com/MinJunKang",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)