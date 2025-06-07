from enum import Enum
from typing import Any

import juliacall
import numpy as np
import scipy.sparse as sp

_jl = juliacall.newmodule("SMCJuliaCall")
_jl.seval("using SparseArrays")
_jl.seval("using SparseMatrixColorings")
_jl.seval("using PythonCall")
_jl.seval("using StableRNGs")


class Structure(Enum):
    SYMMETRIC = "symmetric"
    NONSYMMETRIC = "nonsymmetric"


class Partition(Enum):
    COLUMN = "column"
    ROW = "row"


class Order(Enum):
    NATURAL = "natural"
    RANDOM = "random"
    LARGEST_FIRST = "largest_first"
    SMALLEST_LAST = "smallest_last"
    INCIDENCE_DEGREE = "incidence_degree"
    DYNAMIC_LARGEST_FIRST = "dynamic_largest_first"


def _ColoringProblem(structure: Structure, partition: Partition) -> Any:
    return _jl.ColoringProblem(
        structure=_jl.Symbol(structure.value), partition=_jl.Symbol(partition.value)
    )


def _GreedyColoringAlgorithm(order: Order, seed: int | None) -> Any:
    match order:
        case Order.NATURAL:
            jl_order = _jl.NaturalOrder()
        case Order.RANDOM:
            jl_order = _jl.RandomOrder(_jl.StableRNG(0), seed)
        case Order.LARGEST_FIRST:
            jl_order = _jl.LargestFirst()
        case Order.SMALLEST_LAST:
            jl_order = _jl.SmallestLast()
        case Order.INCIDENCE_DEGREE:
            jl_order = _jl.IncidenceDegree()
        case Order.DYNAMIC_LARGEST_FIRST:
            jl_order = _jl.DynamicLargestFirst()
    return _jl.GreedyColoringAlgorithm(
        jl_order, decompression=_jl.Symbol("direct"), postprocessing=False
    )


def _SparseMatrixCSC(A: np.ndarray[tuple[int, int], Any] | sp.spmatrix) -> Any:
    S = sp.csc_matrix(A)
    m, n = S.get_shape()
    colptr_jl = _jl.pyconvert(_jl.Vector, S.indptr + 1)
    rowval_jl = _jl.pyconvert(_jl.Vector, S.indices + 1)
    nzval_jl = _jl.ones(_jl.Bool, S.nnz)
    S_jl = _jl.SparseMatrixCSC(
        m,
        n,
        colptr_jl,
        rowval_jl,
        nzval_jl,
    )
    return S, S_jl


def compute_coloring(
    sparsity_pattern: np.ndarray[tuple[int, int], Any] | sp.spmatrix,
    structure: Structure = Structure.NONSYMMETRIC,
    partition: Partition = Partition.COLUMN,
    order: Order = Order.NATURAL,
    order_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[sp.csc_matrix, sp.csc_matrix]]:
    """Compute the coloring of a given sparsity pattern and prepare for sparse differentiation.

    Args:
        sparsity_pattern: The sparsity pattern to color. Its values do not matter, only the locations of its nonzero entries.
        structure: The structure of the underlying matrix.
        partition: The partition chosen for coloring.
        order: The order applied to the vertices during greedy coloring.
        order_seed: For the random order, an optional RNG seed to guarantee reproducible execution.

    Returns:
        A tuple `(colors, basis_matrix, (row_inds, col_inds))` where
            - `colors` is a vector of integer colors, numbered from 1.
            - `basis_matrix` gives the column- or row-basis vectors used during compression.
            - `(row_inds, col_inds)` is a tuple of sparse matrices telling at which row and column of the compressed matrix each coefficient should be retrieved.

        These objects can be used within `pysparsematrixcolorings.compress` and `pysparsematrixcolorings.decompress`.
    """
    M, N = sparsity_pattern.shape
    S, S_jl = _SparseMatrixCSC(sparsity_pattern)
    problem_jl = _ColoringProblem(structure, partition)
    algorithm_jl = _GreedyColoringAlgorithm(order, seed=order_seed)
    result_jl = _jl.coloring(S_jl, problem_jl, algorithm_jl)

    match partition:
        case Partition.COLUMN:
            colors = np.array(_jl.column_colors(result_jl)) - 1
        case Partition.ROW:
            colors = np.array(_jl.row_colors(result_jl)) - 1

    compressed_indices = np.array(result_jl.compressed_indices) - 1
    C = np.max(colors)

    match partition:
        case Partition.COLUMN:
            basis_matrix = np.column_stack([colors == c for c in range(C + 1)])
            compressed_row_inds = sp.csc_matrix(
                (compressed_indices % M, S.indices, S.indptr), shape=(M, N)
            )
            compressed_col_inds = sp.csc_matrix(
                (compressed_indices // M, S.indices, S.indptr), shape=(M, N)
            )
        case Partition.ROW:
            basis_matrix = np.vstack([colors == c for c in range(C + 1)])
            compressed_row_inds = sp.csc_matrix(
                (compressed_indices % (C + 1), S.indices, S.indptr), shape=(M, N)
            )
            compressed_col_inds = sp.csc_matrix(
                (compressed_indices // (C + 1), S.indices, S.indptr), shape=(M, N)
            )

    return colors, basis_matrix, (compressed_row_inds, compressed_col_inds)
