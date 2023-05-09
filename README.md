# timings
comparing timings of fortran julia numba numpy

compiling Fortran:
```
f2py -c -m matmul_fort matmul.f90
```

run:
```
python-jl matrix_compare.py
```

requirements:
 - conda
 - julia
 - MKL
 - numpy
 - numba
