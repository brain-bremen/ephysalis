"""Readers of the lab's historic recordings, to be removed once those datasets are converted:

* :mod:`~ephysalis.formats.input.legacy.daq` — the lab's DAQ program (``.daq``, ADC64/MC-128, 1999-2006)
  and the calibration of its recordings;
* :mod:`~ephysalis.formats.input.legacy.uff` — DataWave Experimenter's Workbench (``.UFF``, 1999);
* :mod:`~ephysalis.formats.input.legacy.mcd` — Multi Channel Systems MC_Rack (``.mcd``, 2010-2011).

Their trials come from :mod:`ephysalis.stimulus.legacy`. The rest of ephysalis never imports
either package (``tests/test_legacy_isolation.py``): the modules plug in only through the entry
points in ``pyproject.toml`` (``daq``, ``uff``, ``mcd``, ``vstim_td01``) and the generic interfaces
(:class:`~ephysalis.formats.input.base.InputFormat`, :class:`~ephysalis.stimulus.base.StimulusSystem`
incl. ``run_start``). Removing them = deleting both packages, ``tests/legacy/`` and the four entry points.
"""
