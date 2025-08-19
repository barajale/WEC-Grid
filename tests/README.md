# WEC-GRID Pytest Test Suite

A clean, simple pytest-based test suite for WEC-GRID core functionality.

## Quick Start

```bash
# Install dependencies
pip install -r tests/requirements.txt

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_imports.py -v
```

## Test Structure

- **`test_imports.py`** - Test core module imports
- **`test_grid_state.py`** - Test GridState data structure
- **`test_database.py`** - Test database functionality
- **`test_time_manager.py`** - Test time management
- **`test_plotting.py`** - Test plotting capabilities
- **`conftest.py`** - Pytest configuration and fixtures
- **`pytest.ini`** - Pytest settings

## Features

- ✅ Clean pytest syntax (no classes required)
- ✅ Automatic test discovery
- ✅ Shared fixtures for common test data
- ✅ Graceful handling of missing external dependencies
- ✅ Configured for non-interactive matplotlib backend

## Example Output

```
================================ test session starts =================================
collected 12 items

test_database.py::test_database_creation PASSED                            [  8%]
test_database.py::test_database_connection PASSED                          [ 16%]
test_grid_state.py::test_grid_state_creation PASSED                        [ 25%]
test_imports.py::test_database_import PASSED                               [ 33%]
test_imports.py::test_engine_import PASSED                                 [ 41%]
...

======================== 12 passed in 2.34s ===============================
```
