from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from binding import ModelLib


@dataclass
class DenseReluModel:
    """
    High-level model interface (user-friendly).
    Internally uses ModelLib (ctypes) to call the C function dense_relu.
    """
    in_features: int
    out_features: int
    W: np.ndarray  # flat (out*in,) float32
    b: np.ndarray  # (out,) float32
    lib: ModelLib

    @classmethod
    def from_arrays(
        cls,
        in_features: int,
        out_features: int,
        W: np.ndarray,
        b: np.ndarray,
        dll_path: Path | None = None,
    ) -> "DenseReluModel":
        lib = ModelLib(dll_path=dll_path)

        W = np.asarray(W, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)

        if W.size != in_features * out_features:
            raise ValueError(f"W size mismatch: expected {in_features*out_features}, got {W.size}")
        if b.size != out_features:
            raise ValueError(f"b size mismatch: expected {out_features}, got {b.size}")

        return cls(
            in_features=in_features,
            out_features=out_features,
            W=np.ascontiguousarray(W, dtype=np.float32),
            b=np.ascontiguousarray(b, dtype=np.float32),
            lib=lib,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        x: shape (in_features,) float32
        returns: y shape (out_features,) float32
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size != self.in_features:
            raise ValueError(f"x size mismatch: expected {self.in_features}, got {x.size}")

        return self.lib.dense_relu(
            x=x,
            W=self.W,
            b=self.b,
            in_features=self.in_features,
            out_features=self.out_features,
        )

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        X: shape (batch, in_features)
        returns: Y shape (batch, out_features)
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.in_features:
            raise ValueError(f"X must be shape (batch, {self.in_features}) got {X.shape}")

        Y = np.zeros((X.shape[0], self.out_features), dtype=np.float32)
        for i in range(X.shape[0]):
            Y[i] = self.predict(X[i])
        return Y
