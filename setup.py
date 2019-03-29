#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as req:
    requires = req.read().split("\n")

setuptools.setup(
    name="l2-bayes-opt",
    version="0.0.1",
    author="Anders Kirk Uhrenholt",
    author_email="akuhren@gmail.com",
    description="Bayesian optimization for target vector estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akuhren/target_vector_estimation",
    packages=setuptools.find_packages(),
    install_requires=requires,
    python_requires=">=3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
