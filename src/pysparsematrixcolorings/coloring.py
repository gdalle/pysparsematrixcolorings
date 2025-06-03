import juliacall
import numpy as np
import scipy.sparse as sp

_jl = juliacall.newmodule("SMCJuliaCall")
_jl.seval("using SparseArrays")
_jl.seval("using SparseMatrixColorings")
_jl.seval("using PythonCall")


def _ColoringProblem(structure: str, partition: str):
    return _jl.ColoringProblem(
        structure=_jl.Symbol(structure), partition=_jl.Symbol(partition)
    )


def _GreedyColoringAlgorithm(order: str):
    match order:
        case "natural":
            jl_order = _jl.NaturalOrder()
        case "largestfirst":
            jl_order = _jl.LargestFirst()
        case _:
            raise ValueError("The provided order is invalid")
    return _jl.GreedyColoringAlgorithm(
        jl_order, decompression=_jl.Symbol("direct"), postprocessing=False
    )


def _SparseMatrixCSC(A):
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
    sparsity_pattern,
    structure: str = "nonsymmetric",
    partition: str = "column",
    order: str = "natural",
    return_aux: bool = False,
):
    """Compute the coloring of a given sparsity pattern and prepare for sparse differentiation.

    Args:
        `sparsity_pattern` (matrix-like): The sparsity pattern to color, expressed as a sparse or dense matrix. Its values do not matter, only the locations of its nonzero entries.
        `structure` (`str`, optional): Either `"nonsymmetric`" or `"symmetric"`. Defaults to `"nonsymmetric"`.
        `partition` (`str`, optional): Either `"column`" or `"row"`. Defaults to `"column"`.
        `order` (`str`, optional): Either `"natural`" or `"largestfirst"`. Defaults to `"natural"`.
        `return_aux` (`bool`, optional): Whether to return additional data used during sparse differentiation. Defaults to `False`.

    Raises:
        ValueError: If the options provided are not correct.

    Returns:
    - If `return_aux=False`, a single vector `colors` such that each column or row of the sparsity pattern gets an integer color (depending on the partition).
    - If `return_aux=True`, a tuple `(colors, basis_matrix, (row_inds, col_inds))` where `basis_matrix` gives the column- or row-basis vectors used during compression while `(row_inds, col_inds)` is a tuple of `scipy.sparse.csc_matrix` matrices telling at which row and column of the compressed matrix each coefficient should be retrieved. These objects can be used within `pysparsematrixcolorings.compress` and `pysparsematrixcolorings.decompress`.
    """
    M, N = sparsity_pattern.shape
    S, S_jl = _SparseMatrixCSC(sparsity_pattern)
    problem_jl = _ColoringProblem(structure, partition)
    algorithm_jl = _GreedyColoringAlgorithm(order)
    result_jl = _jl.coloring(S_jl, problem_jl, algorithm_jl)

    match partition:
        case "column":
            colors = np.array(_jl.column_colors(result_jl)) - 1
        case "row":
            colors = np.array(_jl.row_colors(result_jl)) - 1
        case _:
            raise ValueError("The provided partition is invalid")

    if not return_aux:
        return colors
    else:
        compressed_indices = np.array(result_jl.compressed_indices) - 1
        C = np.max(colors)
        match partition:
            case "column":
                basis_matrix = np.column_stack([colors == c for c in range(C + 1)])
                compressed_row_inds = sp.csc_matrix(
                    (compressed_indices % M, S.indices, S.indptr), shape=(M, N)
                )
                compressed_col_inds = sp.csc_matrix(
                    (compressed_indices // M, S.indices, S.indptr), shape=(M, N)
                )
            case "row":
                basis_matrix = np.vstack([colors == c for c in range(C + 1)])
                compressed_row_inds = sp.csc_matrix(
                    (compressed_indices % (C + 1), S.indices, S.indptr), shape=(M, N)
                )
                compressed_col_inds = sp.csc_matrix(
                    (compressed_indices // (C + 1), S.indices, S.indptr), shape=(M, N)
                )

        return colors, basis_matrix, (compressed_row_inds, compressed_col_inds)
