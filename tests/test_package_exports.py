"""The package exports the real implementation of every public function in the mock."""

import inspect

import decision_engine
from decision_engine import mock


def mock_public_functions() -> dict:
    return {name: fn for name, fn in inspect.getmembers(mock, inspect.isfunction) if not name.startswith("_")}


def test_every_mock_function_is_exported_with_identical_signature():
    functions = mock_public_functions()
    assert functions, "mock has no public functions"
    for name, fake in functions.items():
        assert hasattr(decision_engine, name), f"decision_engine does not export {name}"
        real = getattr(decision_engine, name)
        assert inspect.signature(real) == inspect.signature(fake), name


def test_exports_are_the_real_implementations():
    assert set(decision_engine.__all__) == set(mock_public_functions())
    for name in decision_engine.__all__:
        module = getattr(decision_engine, name).__module__
        assert module.startswith("decision_engine.") and module != "decision_engine.mock", (name, module)
