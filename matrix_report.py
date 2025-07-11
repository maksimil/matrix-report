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

LIMIT_SCATTER = 100_000  # Max nnz to make the scatter plot
LIMIT_UMFPACK_COND = 500_000  # Max n to compute condition number using umfpack method
LIMIT_DENSE_COND = 10_000  # Max n to compute condition number using dense methods

NORMALIZE_TOL = 1e-16
GREEN_TOL = 1e-16  # Max abs value to be green
YELLOW_TOL = 0  # Max abs value to be yellow

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


def ComputeCond(mat_coo):
    mat_dense = mat_coo.todense()
    return np.linalg.cond(mat_dense, p=1)


def Normalize(x, mx):
    xscale = math.log10(x / NORMALIZE_TOL + 1)
    mxscale = math.log10(mx / NORMALIZE_TOL + 1)

    if mxscale == 0:
        return 0
    else:
        return max(xscale / mxscale, NORMALIZE_TOL)


def SaveImage(filepath, pos_array, neg_array, type_array, h, w):
    pos_max = pos_array.max()
    neg_max = neg_array.max()

    rgb_array = np.zeros((h, w, 3))

    for i in range(h):
        for j in range(w):
            posv = pos_array[i][j]
            negv = neg_array[i][j]
            rf = Normalize(posv, pos_max)
            bf = Normalize(negv, neg_max)

            if type_array[i][j] == 0:
                rgb_array[i][j][0] = 1
                rgb_array[i][j][1] = 1
                rgb_array[i][j][2] = 1
            elif type_array[i][j] == 1:
                rgb_array[i][j][0] = 1
                rgb_array[i][j][1] = 1
            elif type_array[i][j] == 2:
                rgb_array[i][j][1] = 1
            else:
                rgb_array[i][j][0] = rf
                rgb_array[i][j][2] = bf

    plt.imsave(filepath, rgb_array)


def SaveImagePoints(filepath, mat_coo):
    fig, ax = plt.subplots(figsize=(8, 8))

    n, m = mat_coo.shape

    pos_max = mat_coo.max()
    neg_max = mat_coo.min()

    def PointColor(x):
        ALPHA = 0.2

        if abs(x) <= YELLOW_TOL:
            return (1, 1, 0, ALPHA)

        if abs(x) <= GREEN_TOL:
            return (0, 1, 0, ALPHA)

        if x > 0:
            return (Normalize(x, pos_max), 0, 0, ALPHA)
        else:
            return (0, 0, Normalize(-x, -neg_max), ALPHA)

    indices = list(range(mat_coo.getnnz()))
    random.shuffle(indices)

    rows = [mat_coo.coords[0][k] for k in indices]
    cols = [mat_coo.coords[1][k] for k in indices]
    vals = [mat_coo.data[k] for k in indices]

    ax.scatter(
        cols,
        rows,
        10,
        color=[PointColor(x) for x in vals],
    )

    ax.grid()
    MARGIN_EPS = 0.01
    ax.set_xlim([-MARGIN_EPS * (m - 1), (1 + MARGIN_EPS) * (m - 1)])
    ax.set_ylim([-MARGIN_EPS * (n - 1), (1 + MARGIN_EPS) * (n - 1)])
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ExampleLine(out_dir):
    pos_array = np.zeros((MAX_IMAGE, MAX_IMAGE))
    neg_array = np.zeros((MAX_IMAGE, MAX_IMAGE))

    # 0 is not present, 1 is below YELLOW_TOL, 2 is below GREEN_TOL, 3 is actually present
    type_array = np.zeros((MAX_IMAGE, MAX_IMAGE), dtype=np.uint8)

    for i in range(MAX_IMAGE):
        for j in range(MAX_IMAGE):
            jvalue = 10 ** (40 * j / MAX_IMAGE - 20)
            ivalue = 10 ** (40 * i / MAX_IMAGE - 20)
            pos_array[i][j] = jvalue if jvalue > GREEN_TOL else 0
            neg_array[i][j] = ivalue if ivalue > GREEN_TOL else 0

            if ivalue <= YELLOW_TOL and jvalue <= YELLOW_TOL:
                type_array[i][j] = 1
            elif ivalue <= GREEN_TOL and jvalue <= GREEN_TOL:
                type_array[i][j] = 2
            else:
                type_array[i][j] = 3

    imgpath = os.path.join(IMAGE_DIR, "example.png")
    SaveImage(
        os.path.join(out_dir, imgpath),
        pos_array,
        neg_array,
        type_array,
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


def MatrixLine(name, mat_coo, out_dir):
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

    umfpack_cond = None
    if n == m and n <= LIMIT_UMFPACK_COND:
        umfpack_cond = math.nan
        try:
            lu = scipy.sparse.linalg.splu(mat_coo.tocsc())
            u_diagonal = lu.U.diagonal()
            umfpack_cond = max(abs(x) for x in u_diagonal) / min(
                abs(x) for x in u_diagonal
            )
        except Exception as e:
            print(e)

    dense_cond = None
    if n == m and n <= LIMIT_DENSE_COND:
        dense_cond = ComputeCond(mat_coo)

    rows = mat_coo.coords[0]
    cols = mat_coo.coords[1]
    vals = mat_coo.data

    image_height = min(n, MAX_IMAGE)
    image_width = min(m, MAX_IMAGE)
    pos_array = np.zeros((image_height, image_width))
    neg_array = np.zeros((image_height, image_width))
    type_array = np.zeros((image_height, image_width))

    for k in range(nnz):
        i = rows[k]
        j = cols[k]
        v = vals[k]

        image_i = i * image_height // n
        image_j = j * image_width // m

        if abs(v) > GREEN_TOL:
            type_array[image_i][image_j] = 3
            if v > GREEN_TOL:
                pos_array[image_i][image_j] += v
            elif v < -GREEN_TOL:
                neg_array[image_i][image_j] -= v
        elif abs(v) > YELLOW_TOL:
            type_array[image_i][image_j] = max(type_array[image_i][image_j], 2)
        else:
            type_array[image_i][image_j] = max(type_array[image_i][image_j], 1)

    imgpath_px = os.path.join(IMAGE_DIR, name + "-px.png")
    SaveImage(
        os.path.join(out_dir, imgpath_px),
        pos_array,
        neg_array,
        type_array,
        image_height,
        image_width,
    )

    do_scatter = nnz <= LIMIT_SCATTER

    if do_scatter:
        imgpath_pts = os.path.join(IMAGE_DIR, name + "-pts.png")
        SaveImagePoints(os.path.join(out_dir, imgpath_pts), mat_coo)

    return (
        "<tr>"
        + f"<td><tt><b>{name}</b><br/>"
        + f"{n} x {m}, nnz = {nnz} ({sparsity:8.4f}% )<br/>"
        + f"pos range = ({min_pos:11.4e}, {max_pos:11.4e})<br/>"
        + f"neg range = ({min_neg:11.4e}, {max_neg:11.4e})<br/>"
        + (
            f"umfpack_cond = {umfpack_cond:10.4e}<br/>"
            if umfpack_cond is not None
            else "umfpack_cond = skipped<br/>"
        )
        + (
            f"cond (p=1)   = {dense_cond:10.4e}<br/>"
            if dense_cond is not None
            else "cond (p=1)   = skipped<br/>"
        )
        + "</tt></td>"
        + f'<td><img class="pixelated" src="{imgpath_px}" /></td>'
        + (f'<td><img src="{imgpath_pts}"</td>' if do_scatter else "")
        + "</tr>"
    )


def CreateReport(names, mts, out_dir):
    os.makedirs(os.path.join(out_dir, IMAGE_DIR), exist_ok=True)

    print(f"Output dir: {out_dir}, index file: {os.path.join(out_dir, REPORT_NAME)}")
    print(f"Max dense matrix is {float(LIMIT_DENSE_COND**2) / float(2**28):.4}Gb")

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

    for k in range(len(names)):
        name = names[k]
        mat_coo = mts[k]
        print(f"Processing [{k + 1:3}/{len(names):3}] {name}")
        output += MatrixLine(name, mat_coo, out_dir)

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

    names = mat_files
    mats = [LoadMatrix(os.path.join(mat_dir, f)) for f in mat_files]
    CreateReport(names, mats, out_dir)
