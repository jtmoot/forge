"""Forge API package.

Exports OpenMP thread settings sized to the container CPU quota before any
submodule imports torch; see forge.api.services.runtime.
"""

from forge.api.services.runtime import export_thread_env

export_thread_env()
