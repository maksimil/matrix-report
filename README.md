# matrix-report

A python script for automatic creation of html reports on various properties of
sparse matrices. The script can also be imported and used as a module for more
complex reports.

Supported features:
- [x] Display of basic info: dimension, sparsity, bandwidth
- [x] Display of sparsity pattern in different formats
- [x] Computation and plot of singular values
- [x] Computation and plot of the spectrum
- [x] Approximation of condition number for large matrices
- [x] Bounds on cache misses for blocking storage schemes
- [x] Graph display of a matrix
- [x] Automatic choice of whether to do some of the more
computationally-intensive reporting (spectrum, singular values) depending on
problem size
- [x] PDf-friendly formatting of the html file
- [ ] Simple API to generate matrices out of existing ones (transformations)
- [ ] FFT of the matrix to check for better representation
- [ ] Different standard permutations

## Sparsity pattern representation

## Usage

For an example run you can use the command
```sh
python3 matrix_report.py matrices out
```
`out` is the output directory where the html report and generated images will
be saved.
