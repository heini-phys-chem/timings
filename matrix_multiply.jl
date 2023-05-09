using PyCall
using LinearAlgebra

function matrix_multiply_julia(a::Array{Float64,2}, b::Array{Float64,2})
    #c = *(a, b)
    # Get the matrix dimensions
    m, n1 = size(a)
    n2, k = size(b)

    # Check that the matrix dimensions are compatible
    @assert n1 == n2

    # Allocate memory for the result matrix
    c = similar(a, Float64, m, k)
    BLAS.gemm!('N', 'N', 1.0, a, b, 0.0, c)
    return c
end

# Import the numpy module from Python
#numpy = pyimport("numpy")
#
## Define a Python function that calls the Julia function
#@pydef function matrix_multiply_py(a, b)
#    a_jl = convert(Array{Float64}, numpy.array(a))
#    b_jl = convert(Array{Float64}, numpy.array(b))
#    c_jl = matrix_multiply_julia(a_jl, b_jl)
#    return PyObject(c_jl)
#end
