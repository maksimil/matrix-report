# matrix-report

A script for generating html reports of matrices. Report generation computes
ranges of positive and negative values, sparsity percentage and estimates
condition number using the umfpack method.

A visual representation of the sparsity pattern is also displayed in two ways.

**Pixelated image.** An image is generated. If several matrix values lie under
a single pixel its color is defined the following way. If none of the matrix
values are mentioned in sparse matrix description as entries, the pixel is
white. If some of the values are entries but none of the numerical values
exceed `YELLOW_TOL` (0 by default), the pixel is yellow. If some of them exceed
`YELLOW_TOL` but none exceed `GREEN_TOL` (1e-16 by default), the pixel is
green. Otherwise positive and negative values that exceed `GREEN_TOL` are
summed separately, the sums are normalized separately respectively to global
maximums of sum for the whole image. Positive sum defines the red component,
negative sum defines to blue component, green component is zero.

**Scatter image.** A `matplotlib` plot is generated. A scatter plot with
semitransparent dots with colors defined by the same procedure as for the
pixelated image.

For an example run you can use the command
```sh
python3 matrix_report.py matrices out
```
`out` is the output directory where the html report and generated images will
be saved.
