from __future__ import annotations

import ctypes
from pathlib import Path
import numpy as np


# ---------- DLL loading ----------
def find_dll() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    dll_path = project_root / "build" / "model.dll"
    if not dll_path.exists():
        raise FileNotFoundError(
            f"model.dll not found at: {dll_path}\n"
            "Make sure you built the DLL and it's named model.dll under /build."
        )
    return dll_path


def load_library(dll_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(dll_path))
    lib.dense_relu.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # W
        ctypes.POINTER(ctypes.c_float),  # b
        ctypes.POINTER(ctypes.c_float),  # y
        ctypes.c_size_t,                 # in_features
        ctypes.c_size_t,                 # out_features
    ]
    lib.dense_relu.restype = None
    return lib


# ---------- Model Interface ----------
class DenseReluModel:
    def __init__(self, in_features: int, out_features: int, W: np.ndarray, b: np.ndarray, lib: ctypes.CDLL):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lib = lib

        W = np.asarray(W, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)

        expected_w = self.in_features * self.out_features
        if W.size != expected_w:
            raise ValueError(f"W size mismatch: expected {expected_w}, got {W.size}")
        if b.size != self.out_features:
            raise ValueError(f"b size mismatch: expected {self.out_features}, got {b.size}")

        self.W = np.ascontiguousarray(W, dtype=np.float32)
        self.b = np.ascontiguousarray(b, dtype=np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size != self.in_features:
            raise ValueError(f"x size mismatch: expected {self.in_features}, got {x.size}")

        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.zeros((self.out_features,), dtype=np.float32)

        self.lib.dense_relu(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(self.in_features),
            ctypes.c_size_t(self.out_features),
        )
        return y


# ---------- Input parsing ----------
def parse_vector(line: str, in_features: int) -> np.ndarray | None:
    """
    Parse a line like: "1 -2 0.5 3"
    Returns np.ndarray(float32) of shape (in_features,) or None if parse fails.
    """
    parts = line.strip().split()
    if len(parts) != in_features:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    return np.array(vals, dtype=np.float32)


def print_help(in_features: int):
    print("\nCommands:")
    print("  help                Show this help")
    print("  q / quit / exit      Exit")
    print("\nInput format:")
    print(f"  Enter {in_features} numbers separated by spaces, e.g.:")
    print("    1 -2 0.5 3\n")


# ---------- Main (Interactive) ----------
def main():
    dll_path = find_dll()
    lib = load_library(dll_path)

    # Model params (later we can load these from file)
    in_features = 4
    out_features = 3
    W = np.array([
        0.1, 0.2, 0.3, 0.4,
        -0.5, 0.1, 0.0, 0.2,
        0.3, -0.2, 0.8, -0.1
    ], dtype=np.float32)
    b = np.array([0.0, 0.5, -0.2], dtype=np.float32)

    model = DenseReluModel(in_features, out_features, W, b, lib)

    print("✅ Model loaded!")
    print("DLL:", dll_path)
    print_help(in_features)

    while True:
        line = input("> ").strip()
        if not line:
            continue

        cmd = line.lower()
        if cmd in ("q", "quit", "exit"):
            print("bye 👋")
            break
        if cmd == "help":
            print_help(in_features)
            continue

        x = parse_vector(line, in_features)
        if x is None:
            print(f"❌ Invalid input. Type 'help' for examples. (need exactly {in_features} numbers)")
            continue

        y = model.predict(x)
        # nice printing
        y_str = " ".join(f"{v:.6f}" for v in y.tolist())
        print("output:", y_str)


if __name__ == "__main__":
    main()
