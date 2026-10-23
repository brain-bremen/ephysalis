from ephyscon.scaling import scale_to_16_bit_range
import numpy as np
import pytest


def test_scale_to_16_bit_range():
    # Test with a simple array
    np.random.seed(42)

    data = np.random.rand(100) * 37
    scaled_data, factor = scale_to_16_bit_range(data, scale_abs_max_to=32765)

    assert isinstance(scaled_data, np.ndarray)
    assert scaled_data.dtype == np.int16
    assert np.all(scaled_data >= -32768) and np.all(scaled_data <= 32767)

    # Check if the maximum absolute value is scaled correctly
    expected_scaled_data = np.int16(data / factor)
    assert np.array_equal(scaled_data, expected_scaled_data)


def test_scale_to_16_bit_range_errors():
    # Test with an invalid scale_abs_max_to value
    data = np.random.rand(100) * 37
    with pytest.raises(ValueError) as exc_info:
        scale_to_16_bit_range(data, scale_abs_max_to=40000)

    # Test with a negative scale_abs_max_to value
    with pytest.raises(ValueError) as exc_info:
        scale_to_16_bit_range(data, scale_abs_max_to=-10)
