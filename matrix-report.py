import sys
import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
import math

MAX_IMAGE = 256
IMAGE_DIR = "matrix-images"
REPORT_NAME = "index.html"

NORMALIZE_TOL = 1e-16
GREEN_TOL = 1e-16


def Normalize(x, mx):
    xscale = math.log10(x / NORMALIZE_TOL + 1)
    mxscale = math.log10(mx / NORMALIZE_TOL + 1)

    if mxscale == 0:
        return 0
    else:
        return max(xscale / mxscale, NORMALIZE_TOL)


def SaveImage(filepath, pos_array, neg_array, h, w):
    pos_max = pos_array.max()
    neg_max = neg_array.max()

    rgb_array = np.zeros((h, w, 3))

    for i in range(h):
        for j in range(w):
            posv = pos_array[i][j]
            negv = neg_array[i][j]
            rf = Normalize(posv, pos_max)
            bf = Normalize(negv, neg_max)

            if posv + negv == 0:
                rgb_array[i][j][0] = 1
                rgb_array[i][j][1] = 1
                rgb_array[i][j][2] = 1
            elif posv + negv < GREEN_TOL:
                rgb_array[i][j][1] = 1
            else:
                rgb_array[i][j][0] = rf
                rgb_array[i][j][2] = bf

    plt.imsave(filepath, rgb_array)


argvs = list(sys.argv)


if len(argvs) < 3:
    print("USAGE: matrix-report [matrix directory] [out directory]")
    exit(-1)

mat_dir = argvs[1]
out_dir = argvs[2]
mat_files = [
    f
    for f in os.listdir(mat_dir)
    if os.path.isfile(os.path.join(mat_dir, f)) and f.endswith(".mtx")
]

print(
    f"Matrix dir:   {mat_dir}/\nMatrix files: {mat_files}\n"
    + f"Output file:  {os.path.join(out_dir, REPORT_NAME)}"
)

os.makedirs(os.path.join(out_dir, IMAGE_DIR), exist_ok=True)

output = (
    "<html><style>"
    + "img {image-rendering: pixelated; image-rendering: -moz-crisp-edges;}\n"
    + "tt {white-space: pre;}\n"
    + "</style><body><table>"
)

pos_array = np.zeros((MAX_IMAGE, MAX_IMAGE))
neg_array = np.zeros((MAX_IMAGE, MAX_IMAGE))

for i in range(MAX_IMAGE):
    for j in range(MAX_IMAGE):
        pos_array[i][j] = 10 ** (40 * j / MAX_IMAGE - 20)
        neg_array[i][j] = 10 ** (40 * i / MAX_IMAGE - 20)


imgpath = os.path.join(IMAGE_DIR, "example.png")
SaveImage(
    os.path.join(out_dir, imgpath),
    pos_array,
    neg_array,
    MAX_IMAGE,
    MAX_IMAGE,
)

output += (
    "<tr><td><tt><b>Color example</b><br/>"
    + "range 10^-20 - 10^20<br/>"
    + f"green is < {GREEN_TOL:7.1e}</tt></td>"
)
output += f'<td><img src="{imgpath}" ' + 'width="500" border="1" /></td></tr>'

for f in mat_files:
    print(f"Processing {f}")
    mat_path = os.path.join(mat_dir, f)
    mat_coo = scipy.io.mmread(mat_path)

    n, m = mat_coo.shape
    nnz = mat_coo.getnnz()
    sparsity = float(nnz) / (n * m) * 100

    pos_values = list(x for x in mat_coo.data if x > 0)
    neg_values = list(x for x in mat_coo.data if x < 0)
    min_pos, max_pos, min_neg, max_neg = math.nan, math.nan, math.nan, math.nan
    if len(pos_values) > 0:
        min_pos = min(pos_values)
        max_pos = max(pos_values)
    if len(neg_values) > 0:
        min_neg = min(neg_values)
        max_neg = max(neg_values)

    lu = scipy.sparse.linalg.splu(mat_coo.tocsc())
    u_diagonal = lu.U.diagonal()
    umfpack_cond = max(abs(x) for x in u_diagonal) / min(abs(x) for x in u_diagonal)

    rows = mat_coo.coords[0]
    cols = mat_coo.coords[1]
    vals = mat_coo.data

    image_height = min(n, MAX_IMAGE)
    image_width = min(m, MAX_IMAGE)
    pos_array = np.zeros((image_height, image_width))
    neg_array = np.zeros((image_height, image_width))

    for k in range(nnz):
        i = rows[k]
        j = cols[k]
        v = vals[k]

        image_i = math.floor(float(i) / n * image_height)
        image_j = math.floor(float(j) / m * image_width)

        if v > 0:
            pos_array[image_i][image_j] += v
        else:
            neg_array[image_i][image_j] -= v

    imgpath = os.path.join(IMAGE_DIR, f + ".png")
    SaveImage(
        os.path.join(out_dir, imgpath),
        pos_array,
        neg_array,
        image_height,
        image_width,
    )

    output += f"<tr><td><tt><b>{f}</b><br/>"
    output += f"{n} x {m}, nnz = {nnz} ({sparsity:8.4f}% )<br/>"
    output += f"pos range = ({min_pos:11.4e}, {max_pos:11.4e})<br/>"
    output += f"neg range = ({min_neg:11.4e}, {max_neg:11.4e})<br/>"
    output += f"umfpack_cond = {umfpack_cond:10.4e}</tt></td>"
    output += f'<td><img src="{imgpath}" width="500" border="1"/></td></tr>'

output += "</table></body></html>"

with open(os.path.join(out_dir, REPORT_NAME), "w") as f:
    f.write(output)
