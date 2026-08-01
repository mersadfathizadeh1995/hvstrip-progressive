"""Custom log-frequency axis for PyQtGraph HV ratio plots.

Lifted verbatim from the ``geo_figure`` reference package. Displays Hz
values on an axis whose coordinate space is ``log10(f)``, with ticks at
the customary 1-2-5 sequence across decades.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg


class LogFreqAxis(pg.AxisItem):
    """X-axis that displays Hz values from log10-transformed coordinates."""

    _NICE_MAJORS = [1, 2, 5]
    _NICE_MINORS = [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]

    def tickValues(self, minVal, maxVal, size):
        if minVal >= maxVal:
            return []
        if maxVal > 10 or minVal < -5:
            return super().tickValues(minVal, maxVal, size)

        hz_min = max(10 ** minVal, 0.01)
        try:
            hz_max = 10 ** maxVal
        except OverflowError:
            return super().tickValues(minVal, maxVal, size)

        ticks = []
        major_pos = []
        decade = 10 ** int(np.floor(np.log10(hz_min)))
        while decade <= hz_max * 10:
            for m in self._NICE_MAJORS:
                val = m * decade
                if hz_min <= val <= hz_max:
                    major_pos.append(np.log10(val))
            decade *= 10
        if major_pos:
            ticks.append((None, major_pos))

        minor_pos = []
        decade = 10 ** int(np.floor(np.log10(hz_min)))
        while decade <= hz_max * 10:
            for m in self._NICE_MINORS:
                val = m * decade
                lv = np.log10(val)
                if hz_min <= val <= hz_max and lv not in major_pos:
                    minor_pos.append(lv)
            decade *= 10
        if minor_pos:
            ticks.append((None, minor_pos))
        return ticks

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            try:
                hz = 10 ** v
                if hz >= 10:
                    out.append(f"{hz:.0f}")
                elif hz >= 1:
                    out.append(f"{hz:.1f}")
                else:
                    out.append(f"{hz:.2f}")
            except (OverflowError, ValueError):
                out.append("")
        return out
