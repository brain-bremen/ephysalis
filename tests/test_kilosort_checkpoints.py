"""Tests of the resumable Kilosort steps, with the Kilosort step functions replaced by fakes."""

import inspect

import numpy as np
import pytest

kilosort = pytest.importorskip("kilosort")
from kilosort import datashift, spikedetect, template_matching  # noqa: E402

from ephysalis.kilosort_checkpoints import CheckpointStore, checkpoints  # noqa: E402


class FakeKilosort:
    """Kilosort's call structure (the drift step detects spikes too), with random results."""

    def __init__(self, n_spikes=2_000_000):
        self.calls = []
        self.n_spikes = n_spikes
        self.allocated = {}

    def drift(self, ops, bfile, **kwargs):
        self.calls.append("drift")
        spikedetect.run(ops, bfile)  # goes through the wrapper, which must not checkpoint it
        ops["dshift"] = np.random.rand(3)
        return ops, np.zeros((self.n_spikes, 6))

    def detect(self, ops, bfile, **kwargs):
        self.calls.append("spikedetect")
        self.allocated["spikedetect"] = spikedetect.np.zeros((10**6, 6)).shape[0]
        ops["wPCA"] = np.random.rand(2)
        return np.random.rand(500_000, 3), np.random.rand(5, 2), ops

    def extract(self, ops, bfile, U, **kwargs):
        self.calls.append("extract")
        self.allocated["extract"] = (template_matching.np.zeros((10**6, 3)).shape[0],
                                     tuple(template_matching.torch.zeros((10**6, 2, 3)).shape))
        ops["iU"] = np.random.rand(2)
        return np.random.rand(4, 3), np.random.rand(4, 2), ops


@pytest.fixture
def fake(monkeypatch):
    fake = FakeKilosort()
    monkeypatch.setattr(datashift, "run", fake.drift)
    monkeypatch.setattr(spikedetect, "run", fake.detect)
    monkeypatch.setattr(template_matching, "extract", fake.extract)
    return fake


def pipeline(tmp_path, settings=None):
    """The order of calls in Kilosort's run, returning everything the steps produced."""
    data = tmp_path / "data.bin"
    data.write_bytes(b"\0" * 100)
    ops = {"settings": settings or {"nblocks": 1}, "filename": [data]}
    np.random.seed(1)
    ops, _ = datashift.run(ops, None)
    st0, tF0, _ = spikedetect.run(ops, None)
    st, tF, _ = template_matching.extract(ops, None, None)
    return {**ops, "st0": st0, "st": st, "tF": tF, "after": np.random.rand(2)}


def assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_resume_skips_finished_steps_and_reproduces_the_run(tmp_path, fake):
    with checkpoints(tmp_path / "ckpt"):
        full = pipeline(tmp_path)
    assert fake.calls == ["drift", "spikedetect", "spikedetect", "extract"]
    assert datashift.run == fake.drift  # restored

    fake.calls.clear()
    with checkpoints(tmp_path / "ckpt"):
        assert_same(pipeline(tmp_path), full)
    assert fake.calls == []

    # interrupted during extract: only extract runs again, with the same random state
    (tmp_path / "ckpt" / "extract.pt").unlink()
    with checkpoints(tmp_path / "ckpt"):
        assert_same(pipeline(tmp_path), full)
    assert fake.calls == ["extract"]


def test_checkpoints_are_discarded_when_settings_change(tmp_path, fake):
    with checkpoints(tmp_path / "ckpt"):
        pipeline(tmp_path)
    fake.calls.clear()
    with checkpoints(tmp_path / "ckpt"):
        pipeline(tmp_path, settings={"nblocks": 5})
    assert fake.calls == ["drift", "spikedetect", "spikedetect", "extract"]


def test_spike_arrays_are_preallocated_from_the_previous_step(tmp_path, fake):
    with checkpoints(tmp_path / "ckpt"):
        pipeline(tmp_path)
    assert fake.allocated["spikedetect"] == int(2_000_000 * 1.3)  # from the drift step's spike count
    assert fake.allocated["extract"] == (1_250_000, (1_250_000, 2, 3))  # 2.5 x the spikedetect count
    assert not hasattr(template_matching.np, "_capacity")  # modules restored


def test_store_key(tmp_path):
    CheckpointStore(tmp_path, "a").save("x", {}, 1)
    assert CheckpointStore(tmp_path, "a").has("x")
    assert not CheckpointStore(tmp_path, "b").has("x")


@pytest.mark.parametrize("function", [spikedetect.run, template_matching.extract])
def test_kilosort_allocates_spike_arrays_as_expected(function):
    """The preallocation relies on Kilosort starting its spike arrays at 10**6 rows."""
    source = inspect.getsource(inspect.unwrap(function))
    assert "zeros((10**6," in source
