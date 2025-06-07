from typing import Any

import numpy as np
import scipy.sparse as sp

from .coloring import Partition


def compress(
    matrix: np.ndarray[tuple[int, int], Any] | sp.spmatrix,
    basis_matrix: np.ndarray[tuple[int, int], Any],
    partition: Partition,
) -> np.ndarray[tuple[int, int], Any]:
    """Compress a matrix by multiplying its columns or rows with a set of basis vectors (the inverse operation is `pysparsematrixcolorings.decompress`).

    Args:
        `matrix`: Matrix to compress.
        `basis_matrix`: Matrix containing sums of basis vectors, either in its rows or in its columns.
        `partition`: Specify whether the compression is based on matrix-vector (column) or vector-matrix (row) products.

    Returns:
        ArrayLike: Either `matrix @ basis_matrix` (for a column partition) or `basis_matrix @ matrix` (for a row partition)
    """
    match partition:
        case Partition.COLUMN:
            return matrix @ basis_matrix
        case Partition.ROW:
            return basis_matrix @ matrix
        case _:  # to appease ty
            raise ValueError("Partition not valid")


def decompress(
    compressed_matrix: np.ndarray[tuple[int, int], Any],
    compressed_row_inds: sp.csc_matrix,
    compressed_col_inds: sp.csc_matrix,
) -> sp.csc_matrix:
    """Recover the original matrix from its compressed counterpart (the inverse operation is `pysparsematrixcolorings.compress`).

    Args:
        `compressed_matrix` (ArrayLike): The column- or row-wise compressed matrix.
        `compressed_row_inds`: A sparse matrix with integer values giving the row indices of the compressed matrix associated with each uncompressed coefficient.
        `compressed_col_inds`: A sparse matrix with integer values giving the column indices of the compressed matrix associated with each uncompressed coefficient.

    Returns:
        `scipy.sparse.csc_matrix`: The uncompressed matrix in CSC format.
    """
    linear_row_inds = compressed_row_inds.data
    linear_col_inds = compressed_col_inds.data
    data = compressed_matrix[linear_row_inds, linear_col_inds]
    return sp.csc_matrix(
        (data, compressed_row_inds.indices, compressed_row_inds.indptr),
        shape=compressed_row_inds.shape,
    )
