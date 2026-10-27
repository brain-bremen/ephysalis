import h5py
import numpy as np

from ephysalis.matfile import is_v73, read_mat, read_mat_path
from conftest import write_v73


def _set_fields_attr(group, names):
    """Write MATLAB_fields the way MATLAB does: a vlen array of single characters per name."""
    fields = np.empty(len(names), dtype=object)
    for i, name in enumerate(names):
        fields[i] = np.array(list(name), dtype="S1")
    group.attrs.create("MATLAB_fields", fields, dtype=h5py.vlen_dtype(np.dtype("S1")))


def test_v73_struct_field_order_and_types(tmp_path):
    path = tmp_path / "test.mat"
    write_v73(path, {"s": {"zeta": np.array([1.0, 2.0]), "alpha": np.array([3.0])}})
    with h5py.File(path, "r+") as f:
        name = f["s"].create_dataset("name", data=np.array([[ord(c)] for c in "V1"], dtype=np.uint16))
        name.attrs["MATLAB_class"] = np.bytes_("char")
        flag = f["s"].create_dataset("flag", data=np.array([[1]], dtype=np.uint8))
        flag.attrs["MATLAB_class"] = np.bytes_("logical")
        _set_fields_attr(f["s"], ["zeta", "alpha", "name", "flag"])

    assert is_v73(path)
    s = read_mat(path)["s"]
    assert list(s) == ["zeta", "alpha", "name", "flag"]  # not alphabetical
    np.testing.assert_array_equal(s["zeta"], [1.0, 2.0])
    assert s["alpha"] == 3.0
    assert s["name"] == "V1"
    assert s["flag"] is True
    assert read_mat_path(path, "s/alpha") == 3.0


def test_missing_ok(tmp_path):
    path = tmp_path / "test.mat"
    write_v73(path, {"a": {"x": np.array([1.0])}})
    assert set(read_mat(path, ["a", "b"], missing_ok=True)) == {"a"}
