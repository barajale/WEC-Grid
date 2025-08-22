from pathlib import Path
from setuptools import setup, find_packages

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="wecgrid",
    version="0.1.0",
    author="Alexander Barajas-Ritchie",
    author_email="barajale@oregonstate.edu",
    description="WEC-Grid: A tool for integrating Wave Energy Converter models into power system simulations",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/acep-uaf/WEC-GRID",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={
        "wecgrid": [
            "util/*.json",
            "modelers/wec_sim/*.json",
        ]
    },
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "pypsa",
        "pypower>=5.1.17",
        "grg-pssedata",
        "tqdm>=4.0",
        "requests>=2.0",
        "networkx>=2.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Energy",
    ],
    python_requires=">=3.7",
)
