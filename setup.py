from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

NAME = "ts-learn"
VERSION = "0.0.1"
DESCRIPTION = (
    "A library for modeling and analyzing data using time-series strategies."
)
AUTHOR = "ya boy"
AUTHOR_EMAIL = "andablo@usc.edu"
URL = None
LONG_DESC = (here / "README.md").read_text(encoding="utf-8")
LONG_DESC_CONTYPE = "text/markdown"

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    long_description=LONG_DESC,
    long_description_content_type=LONG_DESC_CONTYPE,
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.9, <4",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # do we need a license if we are not distributing?
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 ",
        "Topic :: Scientific/Engineering :: Machine Learning",
        "Topic :: Scientific/Engineering :: Python Modules",
    ],
)


"""

How to install this package:

    If you already have an environment set up:
        1. Open a terminal using bash
        2. conda activate <env_name>

    If you do not have an environment set up:
        1. Open a terminal using bash
        2. conda create -n <env_name> python=3.9
        3. conda activate <env_name>
        
    1. pip install -e .
        - the -e flag will install the package in editable mode, meaning that changes to the
            source code will be reflected in the installed package
        - the . indicates that the setup.py file is in the current directory
        - the -e flag is optional, but it is recommended for development

How to uninstall this package:
    1. pip uninstall lobkit-learn


Requirements:

    Method 1: pip freeze (ALL PACKAGES)
        - pip freeze will provide us with an exhaustive list of all packages that are installed in
            our environment, not just the ones that are required for our project

    Method 2: Use pipreqs (MINIMAL REQUIREMENTS)
        - using pipreqs will give us the minimal set of packages that are required for our project

        Steps:
            1. pip install pipreqs
            2. pipreqs /path/to/project -
                example: /home/isaiah.andablo/repos/lobkit-learn

        note:
            running pipreqs will create a requirements.txt file in the current directory.
            This file will contain a line `lobster.egg==info` which should be updated to `lobster==1.0.39`
            In the future this will change as we will no longer be using all of lobster as a dependency, but
            instead we will be using its submodules as dependencies
"""