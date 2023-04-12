#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ["ermshaua"]

from pathlib import Path

import toml
from setuptools import find_packages, setup

pyproject = toml.load("pyproject.toml")

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
    
"""Set up package."""
setup(
    author_email=pyproject["project"]["authors"][0]["email"],
    author=pyproject["project"]["authors"][0]["name"],
    classifiers=pyproject["project"]["classifiers"],
    description=pyproject["project"]["description"],
    install_requires=pyproject["project"]["dependencies"],
    include_package_data=True,
    keywords=pyproject["project"]["keywords"],
    license=pyproject["project"]["license"],
    long_description=long_description,
    long_description_content_type='text/markdown',
    name=pyproject["project"]["name"],
    package_data={
        "claspy": [
            "*.csv",
            "*.csv.gz",
            "*.txt",
        ]
    },
    packages=find_packages(
        where=".",
        exclude=["data", "tests", "notebooks"],
    ),
    project_urls=pyproject["project"]["urls"],
    python_requires=pyproject["project"]["requires-python"],
    setup_requires=pyproject["build-system"]["requires"],
    url=pyproject["project"]["urls"]["repository"],
    version=pyproject["project"]["version"],
    zip_safe=False,
)