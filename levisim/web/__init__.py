"""
Interactive Three.js optical-bench web interfaces backed by ``levisim``.

Serves the static 3D web laboratories (which also run standalone on GitHub Pages
with an in-browser physics engine verified against ``levisim``) alongside a
lightweight local HTTP API that computes spectra directly with ``Trap``,
``SpectralDensityTheory`` and ``Simulation``.
"""

from levisim.web.server import (
    STATIC_DIR,
    compute_tweezer_spectra,
    create_server,
    dispatch_request,
    main,
    run_tweezer_simulation,
    serve,
)

__all__ = [
    'STATIC_DIR',
    'compute_tweezer_spectra',
    'create_server',
    'dispatch_request',
    'main',
    'run_tweezer_simulation',
    'serve',
]
