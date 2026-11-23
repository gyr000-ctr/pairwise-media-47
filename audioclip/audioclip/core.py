"""Soft clipping with 4x oversampled true-peak protection.

Public API
----------
``SoftClip(drive=1.0)``
    Construct a stateful clipper. ``drive`` must be a finite positive number
    or :class:`ValueError` is raised.

``SoftClip.process(samples)``
    Process a finite sequence of real samples and return a new ``list`` of
    ``float``. Filter state is carried across calls so splitting a stream into
    buffers yields bit-identical output to processing it in one buffer.
    Non-finite samples (NaN / +-inf) raise :class:`NonFiniteSampleError`,
    distinct from the :class:`ValueError` used for an illegal ``drive``.

Implementation
--------------
The hot loop is a small C shared library (``_native.c``) compiled once on
first use and loaded through :mod:`ctypes` (standard library only). If no C
compiler is available, an equivalent pure-Python implementation is used
automatically; both share the same coefficients and produce the same values.
"""

from __future__ import annotations

import array
import ctypes
import math
import os
import subprocess
import sys
import sysconfig
import tempfile
import threading

__all__ = ["SoftClip", "NonFiniteSampleError", "InvalidDriveError"]


class InvalidDriveError(ValueError):
    """Raised when ``drive`` is not a finite, strictly positive number."""


class NonFiniteSampleError(ValueError):
    """Raised when an input sample is NaN or infinite."""


# --------------------------------------------------------------------------
# DSP constants (must match _native.c)
# --------------------------------------------------------------------------
_OVERSAMPLE = 4
_NTAP = 16
_LINEAR = 0.5
_L4 = 0.985
_LOUT = 0.975
_W0 = 0.93
_R = 0.72


def _build_decimator():
    tail = [_R ** (k + 1) for k in range(_NTAP - 1)]
    s = sum(tail)
    dec = [_W0] + [(1.0 - _W0) * t / s for t in tail]
    return dec


_DEC = _build_decimator()


def _curve(x):
    """Odd, non-decreasing soft saturation; identity for |x| <= _LINEAR."""
    ax = abs(x)
    if ax <= _LINEAR:
        return x
    v = 1.0 - (1.0 - _LINEAR) * math.exp(
        -(ax - _LINEAR) / (1.0 - _LINEAR)
    )
    if v > _L4:
        v = _L4
    return -v if x < 0.0 else v


# --------------------------------------------------------------------------
# Native library loading (compiled once, cached on disk; process-wide lock)
# --------------------------------------------------------------------------
_LIB = None
_LIB_LOCK = threading.Lock()


def _lib_name():
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    # EXT_SUFFIX is an extension suffix (e.g. .cpython-314-...-gnu.so);
    # a plain shared object only needs the platform .so tail.
    if suffix.endswith(".so"):
        return "libaudioclip_native.so"
    return "libaudioclip_native" + suffix


def _compile_native():
    """Compile ``_native.c`` into a cached shared library.

    Returns the loaded library or ``None`` if no working compiler is present.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "_native.c")
    if not os.path.exists(src):
        return None

    cache_dir = os.path.join(
        tempfile.gettempdir(), "audioclip_native_%d" % sys.hexversion
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        return None
    lib_path = os.path.join(cache_dir, _lib_name())

    # Reuse an existing build unless the source is newer than the library.
    if not (os.path.exists(lib_path)
            and os.path.getmtime(lib_path) >= os.path.getmtime(src)):
        cc = os.environ.get("CC", "cc")
        cmd = [
            cc, "-O3", "-fPIC", "-shared",
            "-o", lib_path, src, "-lm",
        ]
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        lib = ctypes.CDLL(lib_path)
    except OSError:
        return None

    lib.ac_create.argtypes = [ctypes.c_double]
    lib.ac_create.restype = ctypes.c_void_p
    lib.ac_destroy.argtypes = [ctypes.c_void_p]
    lib.ac_destroy.restype = None
    lib.ac_process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ac_process.restype = ctypes.c_int
    return lib


def _get_native():
    global _LIB
    if _LIB is not None:
        return _LIB if _LIB is not False else None
    with _LIB_LOCK:
        if _LIB is None:
            try:
                _LIB = _compile_native()
            except Exception:
                _LIB = False
            if _LIB is None:
                _LIB = False
    return _LIB if _LIB is not False else None


# Status codes mirrored from _native.c
_AC_OK = 0
_AC_ERR_NONFINITE = 2


class _NativeBackend:
    """ctypes-backed implementation."""

    def __init__(self, drive):
        lib = _get_native()
        handle = lib.ac_create(ctypes.c_double(drive))
        if not handle:
            # ac_create rejects only bad drive; allocation failure is OOM.
            raise InvalidDriveError(
                "drive must be a finite positive number, got %r" % (drive,)
            )
        self._lib = lib
        self._handle = ctypes.c_void_p(handle)

    def process(self, samples):
        # array.array packs at C speed and rejects non-numeric items with
        # TypeError; NaN/Inf pass through and are reported by the native
        # call as AC_ERR_NONFINITE below.
        values = array.array("d", samples)
        n = len(values)
        if n == 0:
            return []
        in_arr = (ctypes.c_double * n).from_buffer(values)
        out_arr = (ctypes.c_double * n)()
        rc = self._lib.ac_process(
            self._handle, in_arr, ctypes.c_size_t(n), out_arr
        )
        if rc == _AC_ERR_NONFINITE:
            raise NonFiniteSampleError(
                "samples must all be finite (NaN/Inf not allowed)"
            )
        if rc != _AC_OK:
            raise MemoryError("native clipper failed")
        return list(out_arr)

    def close(self):
        if self._handle:
            self._lib.ac_destroy(self._handle)
            self._handle = ctypes.c_void_p(0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _PyBackend:
    """Pure-Python reference implementation, numerically matching the C one."""

    def __init__(self, drive):
        self.drive = drive
        self._hist = [0.0] * _NTAP
        self._primed = False

    def process(self, values):
        if not values:
            return []
        # DC-consistent warm start (mirrors ac_process in _native.c).
        if not self._primed:
            c0 = _curve(values[0] * self.drive)
            self._hist = [c0] * _NTAP
            self._primed = True
        hist = self._hist
        out = []
        for x in values:
            g = x * self.drive
            y = 0.0
            for p in range(_OVERSAMPLE):          # zero-order hold x4
                c = _curve(g)
                hist = [c] + hist[:-1]
                if p == 0:                         # zero-delay decimation
                    acc = 0.0
                    for k in range(_NTAP):
                        acc += _DEC[k] * hist[k]
                    if acc > _LOUT:
                        acc = _LOUT
                    elif acc < -_LOUT:
                        acc = -_LOUT
                    y = acc
            out.append(y)
        self._hist = hist
        return out


def _validate_drive(drive):
    try:
        d = float(drive)
    except (TypeError, ValueError, OverflowError):
        raise InvalidDriveError(
            "drive must be a finite positive number, got %r" % (drive,)
        ) from None
    if not math.isfinite(d) or d <= 0.0:
        raise InvalidDriveError(
            "drive must be a finite positive number, got %r" % (drive,)
        )
    return d


class SoftClip:
    """Stateful soft clipper with 4x oversampled true-peak limiting.

    Parameters
    ----------
    drive:
        Strictly positive, finite input gain. Illegal values raise
        :class:`InvalidDriveError` (a :class:`ValueError` subclass).
    """

    def __init__(self, drive=1.0):
        d = _validate_drive(drive)
        # A per-instance lock makes concurrent process() calls safe; state
        # updates are serialized so two threads cannot interleave history.
        self._lock = threading.RLock()
        self._drive = d
        native = _get_native()
        self._native = native is not None
        if native is not None:
            try:
                self._backend = _NativeBackend(d)
            except InvalidDriveError:
                raise
            except Exception:
                self._backend = _PyBackend(d)
                self._native = False
        else:
            self._backend = _PyBackend(d)

    @property
    def drive(self):
        return self._drive

    @property
    def is_native(self):
        """Whether the compiled C backend is in use."""
        return self._native

    def process(self, samples):
        """Process ``samples`` and return a new list of clipped floats.

        The input is never mutated. Raises
        :class:`NonFiniteSampleError` (a :class:`ValueError` subclass) if any
        sample is NaN or infinite.
        """
        with self._lock:
            if self._native:
                # Fast path: the native backend packs and validates at C
                # speed (NaN/Inf -> AC_ERR_NONFINITE -> NonFiniteSampleError).
                return self._backend.process(samples)
            values = [float(v) for v in samples]
            for v in values:
                if not math.isfinite(v):
                    raise NonFiniteSampleError(
                        "samples must all be finite (NaN/Inf not allowed)"
                    )
            return self._backend.process(values)
