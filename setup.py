from setuptools import setup, find_packages

setup(
    name="wecgrid",
    version="0.1.0",
    author="Alexander Barajas-Ritchie",
    author_email="barajale@oregonstate.edu",
    description="WEC-Grid: A tool for integrating Wave Energy Converter models into power system simulations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/acep-uaf/WEC-GRID",
    license="MIT",  # update if different
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.6",
        "pandas>=1.0.4",
        "matplotlib>=3.1.0",
        "seaborn>=0.11.0",
        "pypsa",
        "pypower>=5.1.17",
        "pyrlu>=0.2.1",
        "ipycytoscape>=1.3.3",
        "spectate>=1.0.1",
        "bqplot",
        "pywin32>=228; platform_system=='Windows'",
        "ipykernel>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Energy",
    ],
    python_requires="==3.7.*",
)