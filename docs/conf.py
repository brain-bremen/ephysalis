"""Sphinx configuration of the ephysalis documentation (``uv run --group docs sphinx-build docs docs/_build``)."""

from importlib.metadata import version as _version

project = "ephysalis"
author = "Brain Research Institute, University of Bremen"
release = _version("ephysalis")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["torch", "kilosort"]

html_theme = "furo"
html_title = f"ephysalis {release}"
