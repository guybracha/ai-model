from __future__ import annotations

import ctypes
from ctypes import c_float, c_size_t, POINTER
from pathlib import Path
from dataclasses import dataclass


class ModelLibraryError(RuntimeError):
    pass


def _default_dll_path() -> Path:
    # py/bindings.py -> project_root/build/model.dll
    return Path(__file__).resolve().parents[1] / "build" / "model.dll"


@dataclass(frozen=True)
class DenseReluSignature:
    in_features: int
    out_features: int


class ModelLib:
    """
    Thin wrapper around model.dll using ctypes.
    Exposes: dense_relu(x, W, b) -> y
    """

    def __init__(self, dll_path: Path | None = None):
        self.dll_path = (dll_path or _default_dll_path()).resolve()
        if not self.dll_path.exists():
            raise ModelLibraryError(
                f"Could not find DLL at: {self.dll_path}\n"
                "Build the C code and place model.dll under /build."
            )

        try:
            self._lib = ctypes.CDLL(str(self.dll_path))
        except OSError as e:
            raise ModelLibraryError(f"Failed to load DLL: {self.dll_path}\n{e}") from e

        # Configure function signature once
        try:
            fn = self._lib.dense_relu
        except AttributeError as e:
            raise ModelLibraryError(
                "dense_relu symbol not found in DLL. "
                "Make sure the function is exported (API macro) and compiled into the DLL."
            ) from e

        fn.argtypes = [
            POINTER(c_float),  # x
            POINTER(c_float),  # W
            POINTER(c_float),  # b
            POINTER(c_float),  # y
            c_size_t,          # in_features
            c_size_t,          # out_features
        ]
        fn.restype = None
        self._dense_relu = fn

    def dense_relu(self, x, W, b, *, in_features: int, out_features: int):
        """
        x: np.ndarray shape (in_features,) dtype float32
        W: np.ndarray shape (out_features*in_features,) row-major float32
        b: np.ndarray shape (out_features,) dtype float32
        returns y: np.ndarray shape (out_features,) dtype float32
        """
        import numpy as np

        # --- validation + conversions ---
        x = np.asarray(x, dtype=np.float32)
        W = np.asarray(W, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        if x.ndim != 1 or x.shape[0] != in_features:
            raise ValueError(f"x must be shape ({in_features},) got {x.shape}")

        if b.ndim != 1 or b.shape[0] != out_features:
            raise ValueError(f"b must be shape ({out_features},) got {b.shape}")

        expected_w = out_features * in_features
        if W.ndim != 1 or W.shape[0] != expected_w:
            raise ValueError(f"W must be flat shape ({expected_w},) got {W.shape}")

        # Ensure contiguous memory
        x = np.ascontiguousarray(x, dtype=np.float32)
        W = np.ascontiguousarray(W, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)

        y = np.zeros((out_features,), dtype=np.float32)

        self._dense_relu(
            x.ctypes.data_as(POINTER(c_float)),
            W.ctypes.data_as(POINTER(c_float)),
            b.ctypes.data_as(POINTER(c_float)),
            y.ctypes.data_as(POINTER(c_float)),
            c_size_t(in_features),
            c_size_t(out_features),
        )
        return y
