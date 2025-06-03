from . import _jl
import numpy as np
import scipy.sparse as sp


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


def coloring(
    sparsity_pattern,
    structure: str = "nonsymmetric",
    partition: str = "column",
    order: str = "natural",
):
    """Perform the coloring of a given sparsity pattern.

    Args:
        sparsity_pattern (matrix-like): The sparsity pattern to color, expressed as a sparse or dense matrix. Its values do not matter, only the locations of its nonzero entries.
        structure (str, optional): Either "nonsymmetric" or "symmetric". Defaults to "nonsymmetric".
        partition (str, optional): Either "column" or "row". Defaults to "column".
        order (str, optional): Either "natural" or "largestfirst". Defaults to "natural".

    Returns:
        np.array: A vector of colors starting at zero, one for each column or row of the sparsity pattern.
    """
    S = sp.coo_matrix(sparsity_pattern)
    row_inds = S.coords[0] + 1
    col_inds = S.coords[1] + 1
    S_jl = _jl.sparse(row_inds, col_inds, _jl.ones(_jl.Bool, S.nnz))
    problem_jl = _ColoringProblem(structure, partition)
    algorithm_jl = _GreedyColoringAlgorithm(order)
    colors_jl = _jl.fast_coloring(S_jl, problem_jl, algorithm_jl)
    colors = np.array(colors_jl, dtype=np.int32) - 1
    return colors
