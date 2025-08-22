# Troubleshooting

Common issues and quick fixes when working with WEC-Grid.

## PSS®E Not Found
- Ensure PSS®E is installed and licensed.
- Add the PSS®E Python path to `sys.path` as shown in the [Installation Guide](../install.md#pss®e-configuration).

## MATLAB Engine Errors
- Install the MATLAB Engine API for Python after creating your conda environment.
- Verify that your Python version matches the MATLAB Engine build (Python 3.7).

## ImportError: `wecgrid`
- Activate the `wecgrid` environment.
- Reinstall with `pip install -e .` from the project root.

If problems persist, please open an issue on the [project repository](https://github.com/acep-uaf/WEC-Grid).


