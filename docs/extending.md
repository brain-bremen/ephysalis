# Extending ephyscon

Input formats and stimulus systems are plugins registered through entry points:

```toml
[project.entry-points."ephyscon.input_formats"]
myformat = "mypackage.reader:MyFormat"          # subclass of ephyscon.io.base.InputFormat

[project.entry-points."ephyscon.stimulus_systems"]
mystim = "mypackage.stimulus:MyStimulusSystem"  # subclass of ephyscon.stimulus.base.StimulusSystem
```

Plugins that fail to import are skipped with a warning. The readers in `ephyscon.legacy`
({doc}`legacy/index`) are examples of formats without probe metadata, with several clocks and
with trial records inside the data file.

## Input formats

```{eval-rst}
.. autoclass:: ephyscon.io.base.InputFormat
   :members:
```

## Stimulus systems

See {doc}`stimulus-systems`: how `timing` uses a plugin, the trial table, and a minimal example.

## Development

```bash
uv sync
uv run pytest
uv run mypy src/
uv run --group docs sphinx-build docs docs/_build   # this documentation
```

Tests against a real SpikeGLX session processed by the MATLAB pipeline run when
`EPHYSCON_TEST_SESSION` points to that session folder.
