"""Readers and plugins for the lab's historic recordings, to be removed once those datasets are
converted:

* :mod:`~ephyscon.legacy.daq` — the lab's DAQ program (``.daq``, ADC64/MC-128, 1999-2006);
* :mod:`~ephyscon.legacy.uff` — DataWave Experimenter's Workbench (``.UFF``, 1999);
* :mod:`~ephyscon.legacy.mcd` — Multi Channel Systems MC_Rack (``.mcd``, 2010-2011);
* :mod:`~ephyscon.legacy.vstim_td01` — VStim 1.x trials of DAQ and MC_Rack runs (``TD01`` records or trigger events + ``.tdr``),
  with the run start on the VStim PC's clock (:mod:`~ephyscon.legacy.pcclock`).

The rest of ephyscon never imports this package (``tests/test_legacy_isolation.py``): the
modules plug in only through the entry points in ``pyproject.toml`` (``daq``, ``uff``, ``mcd``,
``vstim_td01``) and the generic interfaces (:class:`~ephyscon.io.base.InputFormat`,
:class:`~ephyscon.stimulus.base.StimulusSystem` incl. ``run_start``). Removing it = deleting this
package, ``tests/legacy/`` and the four entry points.
"""
