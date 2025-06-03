r"""
A Python interface to the Julia package [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl).
"""

import juliacall

_jl = juliacall.newmodule("SMCJuliaCall")
_jl.seval("using SparseArrays")
_jl.seval("using SparseMatrixColorings")
