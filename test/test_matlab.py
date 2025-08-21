def test_matlab():
    """Test MATLAB modules can be imported."""
    import matlab.engine
    assert matlab.engine is not None
    