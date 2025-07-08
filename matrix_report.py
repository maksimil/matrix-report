import sys
import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
import math
import random

MAX_IMAGE = 256
IMG_WIDTH = 500

IMAGE_DIR = "matrix-images"
REPORT_NAME = "index.html"

DO_LU = True
LIMIT_SCATTER = 100_000  # Max nnz to make the scatter plot

NORMALIZE_TOL = 1e-16
GREEN_TOL = 1e-16

SUPPORTED_EXTENSIONS = [".mtx", ".npz"]


def LoadMatrix(path):
    if path.endswith(".mtx"):
        return scipy.io.mmread(path)

    if path.endswith(".npz"):
        data = np.load(path)
        nr = int(data["nrows"][0])
        nc = int(data["ncols"][0])
        return scipy.sparse.csr_matrix(
            (data["values"], data["col_indices"], data["row_ptr"]), shape=(nr, nc)
        ).tocoo()

    return None


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


def SaveImagePoints(filepath, mat_coo):
    fig, ax = plt.subplots(figsize=(8, 8))

    pos_max = mat_coo.max()
    neg_max = mat_coo.min()

    def PointColor(x):
        ALPHA = 0.2

        if abs(x) < GREEN_TOL:
            return (0, 1, 0, ALPHA)

        if x > 0:
            return (Normalize(x, pos_max), 0, 0, ALPHA)
        else:
            return (0, 0, Normalize(-x, -neg_max), ALPHA)

    indices = list(range(mat_coo.getnnz()))
    random.shuffle(indices)

    cols = [mat_coo.coords[0][k] for k in indices]
    rows = [mat_coo.coords[1][k] for k in indices]
    vals = [mat_coo.data[k] for k in indices]

    ax.scatter(
        cols,
        rows,
        10,
        color=[PointColor(x) for x in vals],
    )

    ax.grid()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ExampleLine(out_dir):
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

    return (
        '<tr class="page-break"><td><tt><b>Color example</b><br/>'
        + "range 10^-20 - 10^20<br/>"
        + f"green is < {GREEN_TOL:7.1e}<br/>"
        + f"images are {MAX_IMAGE}x{MAX_IMAGE} pixels</tt></td>"
        + f'<td><img class="pixelated" src="{imgpath}" /></td></tr>'
    )


def MatrixLine(mat_dir, f, out_dir):
    mat_path = os.path.join(mat_dir, f)
    mat_coo = LoadMatrix(mat_path)

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

    umfpack_cond = math.nan
    if n == m and DO_LU:
        try:
            lu = scipy.sparse.linalg.splu(mat_coo.tocsc())
            u_diagonal = lu.U.diagonal()
            umfpack_cond = max(abs(x) for x in u_diagonal) / min(
                abs(x) for x in u_diagonal
            )
        except Exception as e:
            print(e)

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

        image_i = i * image_height // n
        image_j = j * image_width // m

        if v > 0:
            pos_array[image_i][image_j] += v
        else:
            neg_array[image_i][image_j] -= v

    imgpath_px = os.path.join(IMAGE_DIR, f + "-px.png")
    SaveImage(
        os.path.join(out_dir, imgpath_px),
        pos_array,
        neg_array,
        image_height,
        image_width,
    )

    do_scatter = nnz <= LIMIT_SCATTER

    if do_scatter:
        imgpath_pts = os.path.join(IMAGE_DIR, f + "-pts.png")
        SaveImagePoints(os.path.join(out_dir, imgpath_pts), mat_coo)

    return (
        "<tr>"
        + f"<td><tt><b>{f}</b><br/>"
        + f"{n} x {m}, nnz = {nnz} ({sparsity:8.4f}% )<br/>"
        + f"pos range = ({min_pos:11.4e}, {max_pos:11.4e})<br/>"
        + f"neg range = ({min_neg:11.4e}, {max_neg:11.4e})<br/>"
        + f"umfpack_cond = {umfpack_cond:10.4e}</tt></td>"
        + f'<td><img class="pixelated" src="{imgpath_px}" /></td>'
        + (f'<td><img src="{imgpath_pts}"</td>' if do_scatter else "")
        + "</tr>"
    )


def CreateReport(mat_dir, files, out_dir):
    os.makedirs(os.path.join(out_dir, IMAGE_DIR), exist_ok=True)

    output = (
        "<html><style>"
        + f"img {{ width: {IMG_WIDTH}; border: 1px solid black; }}\n"
        + ".pixelated { image-rendering: pixelated; "
        + "image-rendering: -moz-crisp-edges; }\n"
        + "tt {white-space: pre;}\n"
        + "@media print { .page-break { break-after: page; } }\n"
        + "</style><body><table>"
    )

    output += ExampleLine(out_dir)

    for f in files:
        print(f"Processing {f}")
        output += MatrixLine(mat_dir, f, out_dir)

    output += "</table></body></html>"

    with open(os.path.join(out_dir, REPORT_NAME), "w") as f:
        f.write(output)


if __name__ == "__main__":
    argvs = list(sys.argv)

    if len(argvs) < 3:
        print("USAGE: matrix-report [matrix directory] [out directory]")
        exit(-1)

    mat_dir = argvs[1]
    out_dir = argvs[2]
    mat_files = [
        f
        for f in os.listdir(mat_dir)
        if os.path.isfile(os.path.join(mat_dir, f))
        and os.path.splitext(f)[1] in SUPPORTED_EXTENSIONS
    ]

    CreateReport(mat_dir, mat_files, out_dir)
