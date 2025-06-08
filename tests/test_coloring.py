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
    (colors, _, _) = smc.compute_coloring(S, order=smc.Order("random"), order_seed=2)
    assert np.all(colors == 0)


def test_attila():
    S = np.ones((10, 10), dtype=bool)
    (colors, _, _) = smc.compute_coloring(S)
    assert np.all(colors == np.arange(10))


def test_column_compression():
    As = [
        sp.random(m, n, density)
        for (m, n) in [(100, 200), (200, 100)]
        for density in np.arange(0.02, 0.1, 0.02)
    ]
    partition = smc.Partition("column")
    for A in As:
        colors, basis_matrix, (row_inds, col_inds) = smc.compute_coloring(
            A, partition=partition
        )
        B = smc.compress(A, basis_matrix, partition=partition)
        A2 = smc.decompress(B, row_inds, col_inds)
        assert not np.any(A.todense() != A2)


def test_row_compression():
    As = [
        sp.random(m, n, density)
        for (m, n) in [(100, 200), (200, 100)]
        for density in np.arange(0.02, 0.1, 0.02)
    ]
    partition = smc.Partition("row")
    for A in As:
        colors, basis_matrix, (row_inds, col_inds) = smc.compute_coloring(
            A, partition=partition
        )
        B = smc.compress(A, basis_matrix, partition=partition)
        A2 = smc.decompress(B, row_inds, col_inds)
        assert not np.any(A.todense() != A2)


def test_symmetric_compression():
    As = [
        sp.random(m, n, density)
        for (m, n) in [(200, 200)]
        for density in np.arange(0.02, 0.1, 0.02)
    ]
    structure = smc.Structure("symmetric")
    partition = smc.Partition("column")
    for A in As:
        A = A + A.T
        colors, basis_matrix, (row_inds, col_inds) = smc.compute_coloring(
            A, structure=structure
        )
        B = smc.compress(A, basis_matrix, partition=partition)
        A2 = smc.decompress(B, row_inds, col_inds)
        assert not np.any(A.todense() != A2)
