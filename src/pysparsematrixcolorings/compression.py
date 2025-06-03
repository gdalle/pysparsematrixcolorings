import scipy.sparse as sp


def compress(matrix, basis_matrix, partition: str):
    """Compress a matrix by multiplying its columns or rows with a set of basis vectors (the inverse operation is `pysparsematrixcolorings.decompress`).

    Args:
        `matrix` (matrix-like): Matrix to compress.
        `basis_matrix` (matrix-like): Matrix containing sums of basis vectors, either in its rows or in its columns.
        `partition` (`str`): Either `"column"` or "`row"`, to decide whether the compression is based on matrix-vector or vector-matrix products.

    Returns:
        matrix-like: Either `matrix @ basis_matrix` (for a column partition) or `basis_matrix @ matrix` (for a row partition)
    """
    match partition:
        case "column":
            return matrix @ basis_matrix
        case "row":
            return basis_matrix @ matrix


def decompress(compressed_matrix, compressed_row_inds, compressed_col_inds):
    """Recover the original matrix from its compressed counterpart (the inverse operation is `pysparsematrixcolorings.compress`).

    Args:
        `compressed_matrix` (matrix-like): The column- or row-wise compressed matrix.
        `compressed_row_inds` (`scipy.sparse.csc_matrix`): A sparse matrix with integer values giving the row indices of the compressed matrix associated with each uncompressed coefficient.
        `compressed_col_inds` (`scipy.sparse.csc_matrix`): A sparse matrix with integer values giving the column indices of the compressed matrix associated with each uncompressed coefficient.

    Returns:
        `scipy.sparse.csc_array`: The uncompressed matrix in CSC format.
    """
    linear_row_inds = compressed_row_inds.data
    linear_col_inds = compressed_col_inds.data
    data = compressed_matrix[linear_row_inds, linear_col_inds]
    return sp.csc_matrix(
        (data, compressed_row_inds.indices, compressed_row_inds.indptr),
        shape=compressed_row_inds.shape,
    )
