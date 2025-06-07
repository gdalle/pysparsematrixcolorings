import pysparsematrixcolorings as smc
import numpy as np
import scipy.sparse as sp


def test_identity():
    S = sp.eye(10, dtype=bool)
    for order in [
        smc.Order("natural"),
        smc.Order("random"),
        smc.Order("largest_first"),
        smc.Order("smallest_last"),
        smc.Order("incidence_degree"),
        smc.Order("dynamic_largest_first"),
    ]:
        (colors, _, _) = smc.compute_coloring(S, order=order)
        assert np.all(colors == 0)


def test_attila():
    S = np.ones((10, 10), dtype=bool)
    (colors, _, _) = smc.compute_coloring(S)
    assert np.all(colors == np.arange(10))


def test_column_compression():
    A = np.array(
        [
            [0, 2, 3, 0],
            [0, 0, 0, 4],
            [1, 5, 0, 0],
            [0, 6, 7, 0],
        ]
    )
    partition = smc.Partition("column")
    colors, basis_matrix, (row_inds, col_inds) = smc.compute_coloring(
        A, partition=partition
    )
    B = smc.compress(A, basis_matrix, partition=partition)
    A2 = smc.decompress(B, row_inds, col_inds)
    assert not np.any(A != A2)


def test_row_compression():
    A = np.array(
        [
            [0, 2, 3, 0],
            [0, 0, 0, 4],
            [1, 5, 0, 0],
            [0, 6, 7, 0],
        ]
    )
    partition = smc.Partition("row")
    colors, basis_matrix, (row_inds, col_inds) = smc.compute_coloring(
        A, partition=partition
    )
    B = smc.compress(A, basis_matrix, partition=partition)
    A2 = smc.decompress(B, row_inds, col_inds)
    assert not np.any(A != A2)


def test_symmetric_compression():
    A = sp.csc_matrix(sp.diags(np.arange(10)))
    A[:, 0] = np.arange(10, 20)
    A[0, :] = np.arange(10, 20)
    structure = smc.Structure("symmetric")
    partition = smc.Partition("column")
    colors, basis_matrix, (row_inds, col_inds) = smc.compute_coloring(
        A, structure=structure
    )
    B = smc.compress(A, basis_matrix, partition=partition)
    A2 = smc.decompress(B, row_inds, col_inds)
    assert not np.any(A.todense() != A2)
