import numpy as np


def scale_to_16_bit_range(
    data: np.ndarray, scale_abs_max_to: int = 32765
) -> tuple[np.ndarray, float]:
    if scale_abs_max_to <= 0 or scale_abs_max_to > 2**16 / 2:
        raise ValueError(
            f"Integer value to be used for scaling the maximum data value to must be within the 16-bit range (0-{2**16 / 2})"
        )

    max_abs_value_in_data = np.max(np.abs(data))
    factor: float = scale_abs_max_to / max_abs_value_in_data

    scaled_data = (data * factor).astype(np.int16)
    return scaled_data, 1 / factor
