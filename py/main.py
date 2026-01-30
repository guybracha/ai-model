import ctypes
import numpy as np
from pathlib import Path

dll_path = Path(__file__).with_name("model.dll")
lib = ctypes.CDLL(str(dll_path))

# הגדרת חתימה
lib.dense_relu.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.POINTER(ctypes.c_float),  # W
    ctypes.POINTER(ctypes.c_float),  # b
    ctypes.POINTER(ctypes.c_float),  # y
    ctypes.c_size_t,                 # in_features
    ctypes.c_size_t,                 # out_features
]
lib.dense_relu.restype = None

# "פרמטרים של מודל"
in_features = 4
out_features = 3

x = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
W = np.array([
    0.1, 0.2, 0.3, 0.4,
    -0.5, 0.1, 0.0, 0.2,
    0.3, -0.2, 0.8, -0.1
], dtype=np.float32)  # shape (3,4) בשורה רציפה
b = np.array([0.0, 0.5, -0.2], dtype=np.float32)

y = np.zeros(out_features, dtype=np.float32)

lib.dense_relu(
    x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    in_features,
    out_features
)

print("output:", y)
