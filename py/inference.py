from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path

from binding import ModelLib


@dataclass
class DenseReluModel:
    in_features: int
    out_features: int
    W: np.ndarray  # flat (out*in,)
    b: np.ndarray  # (out,)
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
            raise ValueError("W size mismatch")
        if b.size != out_features:
            raise ValueError("b size mismatch")

        return cls(
            in_features=in_features,
            out_features=out_features,
            W=np.ascontiguousarray(W, dtype=np.float32),
            b=np.ascontiguousarray(b, dtype=np.float32),
            lib=lib,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.lib.dense_relu(
            x=x,
            W=self.W,
            b=self.b,
            in_features=self.in_features,
            out_features=self.out_features,
        )


def demo():
    in_features = 4
    out_features = 3

    x = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)

    W = np.array([
        0.1, 0.2, 0.3, 0.4,
        -0.5, 0.1, 0.0, 0.2,
        0.3, -0.2, 0.8, -0.1
    ], dtype=np.float32)

    b = np.array([0.0, 0.5, -0.2], dtype=np.float32)

    model = DenseReluModel.from_arrays(in_features, out_features, W, b)
    y = model.predict(x)

    print("output:", y)


if __name__ == "__main__":
    demo()
