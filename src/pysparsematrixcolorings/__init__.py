import juliacall

jl = juliacall.newmodule("SMCJuliaCall")
jl.seval("using SparseArrays")
jl.seval("using SparseMatrixColorings")
