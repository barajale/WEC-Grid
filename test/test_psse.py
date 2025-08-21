def test_psse():
    """Test PSS®E modules can be imported."""
    import pssepath
    pssepath.add_pssepath()
    import psspy
    import psse35
    assert psspy is not None
    assert psse35 is not None