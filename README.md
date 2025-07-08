# matrix-report

A script for generating html reports of matrices. Report generation computes
ranges of positive and negative values, sparsity percentage and estimates
condition number using the umfpack method.

A visual representation of the sparsity pattern is also displayed. If several
nonzero elements lie in the same pixel, positive and negative parts are summed
separately, red and blue components are computed from positive and negative
sums respectively and mixed to get the resulting color. If both sums are below
a certain tolerance green is displayed instead. An example reference is
provided in each report. Red and blue components are computed using the
`Normalize` function where `x` is the sum value and `mx` is the maximum sum
value.

For an example run you can use the command
```sh
python3 matrix_report.py matrices out
```
`out` is the output directory where the html report and generated images will
be saved.
