r"""
A Python interface to the Julia package [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl).
"""

from .coloring import Partition, Structure, Order, compute_coloring
from .compression import compress, decompress

__all__ = [
    "Partition",
    "Structure",
    "Order",
    "compute_coloring",
    "compress",
    "decompress",
]
