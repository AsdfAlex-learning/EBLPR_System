"""
Lightweight MATLAB Engine client wrapper.

Requires that matlab.engine was installed using MATLAB's provided installer
instructions (i.e. not necessarily via pip). This wrapper exposes start/stop
and a `call_function` helper using MATLAB's `feval`.

The client will add the project's `matlab_core` directory to MATLAB path when
starting so that .m files placed there are callable.
"""
from __future__ import annotations

import os
import threading
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

try:
    # import the matlab.engine module object as matlab_engine for clarity
    import matlab.engine as matlab_engine  # type: ignore
    _HAS_MATLAB = True
except Exception:
    matlab_engine = None  # type: ignore
    _HAS_MATLAB = False


class MatlabClient:
    def __init__(self, matlab_core_dir: Optional[str] = None) -> None:
        self._eng = None
        self._lock = threading.RLock()
        self.matlab_core_dir = matlab_core_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'matlab_core')

    def start(self) -> None:
        """Start MATLAB engine if not already started."""
        if not _HAS_MATLAB:
            raise ImportError('matlab.engine is not available in this Python environment')

        with self._lock:
            if self._eng is None:
                logger.info('Starting MATLAB engine...')
                # start_matlab may take a while; this blocks the calling thread
                self._eng = matlab_engine.start_matlab()
                # add matlab_core directory so user .m files can be called
                try:
                    if os.path.isdir(self.matlab_core_dir):
                        # Pass absolute path
                        self._eng.addpath(self.matlab_core_dir, nargout=0)
                        logger.info('Added matlab_core to MATLAB path: %s', self.matlab_core_dir)
                    else:
                        logger.warning('matlab_core directory does not exist: %s', self.matlab_core_dir)
                except Exception:
                    logger.exception('Failed to add matlab_core to MATLAB path')

    def is_running(self) -> bool:
        return self._eng is not None

    def call_function(self, func_name: str, *args: Any, nargout: int = 1) -> Any:
        """Call a MATLAB function by name using feval.

        Note: simple types (strings, numbers) are passed automatically. For
        arrays you may need to convert to matlab.double in the caller.
        """
        if not _HAS_MATLAB:
            raise ImportError('matlab.engine is not available in this Python environment')

        self.start()

        with self._lock:
            try:
                result = self._eng.feval(func_name, *args, nargout=nargout)
                return result
            except Exception:
                logger.exception('MATLAB function call failed: %s', func_name)
                raise

    def stop(self) -> None:
        """Stop MATLAB engine if running."""
        with self._lock:
            if self._eng is not None:
                try:
                    self._eng.quit()
                except Exception:
                    logger.exception('Error quitting MATLAB engine')
                finally:
                    self._eng = None


# singleton client
client = MatlabClient()
