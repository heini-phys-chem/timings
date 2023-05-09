import time

import numpy as np
from numba import njit
import numba as nb

from julia import Julia
from julia import Main

import matmul_fort


def multiply_matrices_python(a, b):
    m = len(a)
    n = len(b[0])
    p = len(b)
    result = [[0] * n for _ in range(m)]

    for i in range(m):
        for j in range(n):
            for k in range(p):
                result[i][j] += a[i][k] * b[k][j]

    return result

def multiply_matrices_numpy(a, b):
    return np.matmul(a, b)

def multiply_matrices_fortran(a, b):
    m, n = a.shape
    n2, p = b.shape
    assert n == n2, "Incompatible dimensions"
    #c = np.zeros((m, p))
    c = matmul_fort.matrixmul(a, b, m, n)
    return c

@nb.njit
def multiply_matrices_numba(A, B):
    # Get the matrix dimensions
    return np.dot(A.astype(np.float64), B.astype(np.float64))

def matmul_julia(a, b):
    return Main.matrix_multiply_julia(a, b)

def main():
    # Test the function with two 100x100 matrices
    a = [[i+j for j in range(1000)] for i in range(1000)]
    b = [[i-j for j in range(1000)] for i in range(1000)]

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    #start = time.time()
    #result = multiply_matrices_python(a, b)
    #end = time.time()
    #print("Python time: ", end - start)


    start = time.time()
    result_np = multiply_matrices_numpy(a, b)
    end = time.time()
    print("NumPy time:  ", end - start)

    start = time.time()
    Main.include("matrix_multiply.jl")
    end = time.time()
    print("Julia lib time:  ", end - start)

    start = time.time()
    c = matmul_julia(a, b)
    end = time.time()
    print("Julia time:  ", end - start)

    start = time.time()
    c = matmul_julia(a, b)
    end = time.time()
    print("Julia time:  ", end - start)

    #if np.allclose(c, result_np): print("same")

    start = time.time()
    multiply_matrices_fortran(a, b)
    end = time.time()
    print("Fortran time:  ", end - start)

    start = time.time()
    multiply_matrices_numba(a, b)
    end = time.time()
    print("Numba time:  ", end - start)

    start = time.time()
    multiply_matrices_numba(a, b)
    end = time.time()
    print("Numba time:  ", end - start)

if __name__ == '__main__':
    main()
