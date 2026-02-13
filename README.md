# MatrixReport

A python script for automatic creation of html reports on various properties of
sparse matrices. The script can also be imported and used as a module for more
complex reports. Reports are rendered as html which looks like this:

![Picture 1](./readme-figures/example1.png)
![Picture 2](./readme-figures/example2.png)

[ex27-report.pdf](./readme-figures/ex27-report.pdf) is an example of the html
rendered to pdf.

## Features

The following features are currently supported:
- Display of basic info: dimension, sparsity, bandwidth, symmetry, etc.
- Visual display of the sparsity pattern
- Plotting of singular values and the spectrum
- Approximation of the condition number for large matrices
- Bounds on cache misses for different blocking storage schemes
- Histograms for different statistics: block nonzeros, values, row nonzeros and norms
- Graph display of a matrix with 3d animated mode
- Plot Frobenius norm of an error during sparsification with a tolerance
- PDF-friendly formatting of the html output
- High customizability of all the plots
- Parallel processing of matrices
- Lazy-loading of matrices to avoid overwhelming the RAM

The following features are planned:
- Simple API to generate matrices out of existing ones (transformations)
- FFT of the matrix to check for a better representation
- Different standard permutations and transformations

Full docs are located at [DOCS.md](./DOCS.md).

## Usage

`matrix_report.py` can be used as a cli application in the following way:
```sh
python3 matrix_report.py [MATRIX_FOLDER] [OUTPUT_FOLDER]
```
This will read all matrix files from `MATRIX_FOLDER`, process them and write an
html report and its images to `OUTPUT_FOLDER`. To see the report open
`OUTPUT_FOLDER/index.html` in your browser.

For an example run you can use the command
```sh
python3 matrix_report.py matrices out
```

