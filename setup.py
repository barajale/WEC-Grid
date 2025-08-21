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
    include_package_data=True,  # works with MANIFEST.in
    package_data={
        "wecgrid": [
            "data/grid_models/*",
            "data/wec_models/**/*",
            "database/*",
            "modelers/wec_sim/*.m",
        ]
    },
    install_requires=[
        "numpy>=1.21,<1.24",
        "pandas>=1.3,<2.0",
        "matplotlib>=3.4,<3.7",
        "seaborn>=0.11,<0.13",
        # TODO should pin a pyPSA verison 
        "pypsa",
        "pypower>=5.1.17",
        "pyrlu>=0.2.1",
        "ipycytoscape>=1.3.3",
        "spectate>=1.0.1",
        "bqplot",
        # 3.7-only resource loader
        "importlib_resources>=5.12,<6",
        "pywin32>=228; platform_system=='Windows'",
        "ipykernel>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Energy",
    ],
    python_requires="==3.7.*",
)