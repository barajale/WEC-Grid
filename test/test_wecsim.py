"""Tests for WEC-Sim integration."""

import io
import pytest


@pytest.mark.matlab
def test_matlab():
    """Test MATLAB modules can be imported."""
    engine = pytest.importorskip("matlab.engine")
    assert engine is not None


@pytest.mark.wec_sim
def test_wecsim():
    """Test basic WEC-Sim engine initialization."""
    wecgrid = pytest.importorskip("wecgrid")
    pytest.importorskip("matlab.engine")
    test = wecgrid.Engine()
    assert test is not None
    test.wecsim.start_matlab()


@pytest.mark.wec_sim
def test_wecsim_run_from_sim():
    """Test WEC-Sim runFromSimTest functionality."""
    matlab_engine_module = pytest.importorskip("matlab.engine")
    wecgrid = pytest.importorskip("wecgrid")

    # Setup output capture
    out = io.StringIO()
    err = io.StringIO()

    # Initialize engines
    engine = wecgrid.Engine()
    matlab_engine = matlab_engine_module.start_matlab()

    # Change to WEC-Sim path
    matlab_engine.cd(engine.wecsim.get_wec_sim_path())

    # Run the WEC-Sim test
    matlab_engine.eval(
        "wecSimTest(bemioTest=false, regressionTest=false, compilationTest=true, runFromSimTest=true, rotationTest=false)",
        nargout=0,
        stdout=out,
        stderr=err,
    )

    # Get the output
    output = out.getvalue()
    error_output = err.getvalue()

    # Print output for debugging
    print("STDOUT:", output)
    if error_output:
        print("STDERR:", error_output)

    # Assertions to verify the test ran successfully
    assert "Running runFromSimTest" in output
    assert "Done runFromSimTest" in output
    assert "WEC-Sim: An open-source code for simulating wave energy converters" in output
    assert "2 Passed, 0 Failed, 0 Incomplete" in output

    # Check that no errors occurred
    assert error_output == "" or "error" not in error_output.lower()

    # Verify specific test cases passed
    assert "runFromSimTest/fromSimCustom" in output
    assert "runFromSimTest/fromSimInput" in output

    # Clean up
    matlab_engine.quit()

