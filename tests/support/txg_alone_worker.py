# SPDX-License-Identifier: Apache-2.0
"""The worker extension of the qualification driver's ``alone`` mode
(TXG-QUAL-028; core's ``--worker-extension-cls``,
``support.txg_qual_driver.ALONE_READER``).

The ``alone`` mode serves its requests through a multiprocess ``AsyncLLM``,
so the engine core and its worker run in another process. Core imports this
module there when it resolves the worker extension
(``WorkerWrapperBase.init_worker``), after it loads the plugins and before
it builds the model runner, loads the model or runs a capture. The import
installs the driver's startup probes (``txg_qual_driver.StartupProbes``) in
that process; the worker methods below return their readings and the
counters by name through ``AsyncLLM.collective_rpc``. Core's message
serializer sends a method name but refuses a function unless insecure
serialization is on, which the mode does not set. No other process imports
this module. The engine-core process's ``PYTHONPATH`` names ``tests``.
"""

from __future__ import annotations

from typing import Any

from support import txg_qual_driver as d

#: The engine-core process's startup probes, installed on import.
PROBES = d.StartupProbes().install()


class AloneReader:
    """The worker methods of the ``alone`` mode."""

    def alone_counters(self) -> dict[str, Any]:
        """The worker's cell replays and fallbacks by reason
        (``txg_qual_driver.alone_counters``)."""
        return d.alone_counters(self)

    def alone_startup(self, spec: dict[str, Any]) -> dict[str, Any]:
        """The startup probes, the capture record and the run record's
        fields, read after startup (``txg_qual_driver.alone_startup``)."""
        return d.alone_startup(self, spec, PROBES)
