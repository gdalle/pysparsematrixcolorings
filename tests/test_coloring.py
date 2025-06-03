import pysparsematrixcolorings.coloring as smc
import numpy as np
import scipy.sparse as sp


def test_identity():
    sparsity_pattern = sp.eye(10, dtype=bool)
    colors = smc.coloring(sparsity_pattern)
    assert np.all(colors == 0)


def test_attila():
    sparsity_pattern = np.ones((10, 10), dtype=bool)
    colors = smc.coloring(sparsity_pattern)
    assert np.all(colors == np.arange(10))
