import pytest
import pickle
import tempfile
import os


def test_pickle_memoize_warns_on_cache_load():
    """Test that loading a cached pickle file emits a UserWarning."""
    from tvm.contrib.pickle_memoize import memoize

    # Create a cache file
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache")

        @memoize("test_warning_cache")
        def dummy_func():
            return 42

        # First call creates cache
        result = dummy_func()
        assert result == 42

        # Second call loads from cache — should warn
        with pytest.warns(UserWarning, match="Pickle files can execute arbitrary code"):
            result2 = dummy_func()
            assert result2 == 42
