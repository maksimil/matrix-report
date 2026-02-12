import sys
import time
import os
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
import multiprocessing
import math
import dataclasses
from dataclasses import dataclass
from types import NoneType
from typing import Literal
from collections.abc import Callable
import importlib

# -------------------------
# --- Global parameters ---

IMAGE_DIR = "matrix-images"
REPORT_NAME = "index.html"
DEFAULT_PX_COLS = 256
PLOT_MARGIN = 0.01
COL_PER_ROW_MAX = 2
AUTO_INCREASE_INT_SIZE = True
SEED = 42
PLOT_DPI = 200
MIN_PNG_SIZE = 1024
MAX_HIST_BINS = 100

# -----------------------------
# --- Optional dependencies ---

JIT_ENABLED = False
GRAPH_ENABLED = False

# Numba

numba = None


def jit(f):
    pass


if importlib.util.find_spec("numba") is None:
    print("WARN: install numba to accelerate some computations")

    def jit(f):
        return f
else:
    import numba

    JIT_ENABLED = True

    def jit(f):
        return numba.njit(f, cache=True)

# Graph

nx = None
PIL = None
aggdraw = None

if importlib.util.find_spec("networkx") is None:
    print("WARN: install networkx, PIL and aggdraw to use graph layouts")
else:
    import networkx as nx
    import PIL
    import PIL.Image
    import aggdraw

    GRAPH_ENABLED = True

# --------------
# --- Report ---

CSRMatrix = scipy.sparse.csr_matrix


@dataclass
class NamedCSRMatrix:
    name: str
    matrix: CSRMatrix


@dataclass
class NamedMatrixPath:
    name: str
    path: str


MatrixDescType = CSRMatrix | str | NamedCSRMatrix | NamedMatrixPath
MatrixLoader = Callable[[], MatrixDescType]


@dataclass
class ShadeParams:
    enabled: bool = True
    pxRows: int = -1  # scale to ratio
    pxCols: int = DEFAULT_PX_COLS
    zeroTol: float = 0
    scale: Literal["linear"] = "linear"

    def Disabled():
        return ShadeParams(enabled=False)


@dataclass
class ColormapParams:
    scale: Literal["log", "linear"] = "log"
    zeroTol: float = 0
    minColor: float = 1e-12
    maxPosColor: float | NoneType = None
    maxNegColor: float | NoneType = None
    aggRule: Literal["max", "avg"] = "max"

    def SetMax(self, pos, neg):
        if self.maxPosColor is None:
            self.maxPosColor = pos
        if self.maxNegColor is None:
            self.maxNegColor = -neg


def ScaleValue(scale, minv, maxv, v: float):
    if scale == "log":
        vscaled = math.log10(v / minv + 1)
        mscaled = math.log10(maxv / minv + 1)
        return min(vscaled / mscaled, 1)

    if scale == "linear":
        vscaled = (v - minv) / (maxv - minv)
        return np.clip(vscaled, 0, 1)

    return None


def GetValuesColor(values: list[float], colormap: ColormapParams):
    if type(values) is float:
        values = [values]

    pos = [+v for v in values if +v > colormap.zeroTol]
    neg = [-v for v in values if -v > colormap.zeroTol]

    if len(values) == 0:
        return (1, 1, 1)

    if colormap.zeroTol == 0 and len(pos) + len(neg) == 0:
        return (1, 1, 0)  # yellow

    posV, negV = 0, 0
    if len(pos) > 0:
        if colormap.aggRule == "max":
            posV = max(pos)
        elif colormap.aggRule == "avg":
            posV = sum(pos) / len(pos)
        else:
            print(f"WARN: invalid aggRule={colormap.aggRule}")
    if len(neg) > 0:
        if colormap.aggRule == "max":
            negV = max(neg)
        elif colormap.aggRule == "avg":
            negV = sum(neg) / len(neg)
        else:
            print(f"WARN: invalid aggRule={colormap.aggRule}")

    if posV < colormap.minColor and negV < colormap.minColor:
        return (0, 1, 0)  # green

    red = ScaleValue(colormap.scale, colormap.minColor, colormap.maxPosColor, posV)
    blu = ScaleValue(colormap.scale, colormap.minColor, colormap.maxNegColor, negV)

    return (red, 0, blu)


@dataclass
class ColorParams:
    enabled: bool = True
    pxRows: int = -1
    pxCols: int = DEFAULT_PX_COLS
    colormap: ColormapParams = ColormapParams()

    def Disabled():
        return ColorParams(enabled=False)


@dataclass
class ScatterParams:
    enabled: bool = True
    alpha: float = 0.2
    pointSize: float = 10
    colormap: ColorParams = ColormapParams()

    def Disabled():
        return ColorParams(enabled=False)


@dataclass
class HistParams:
    enabled: bool = True
    blockR: int = 32
    blockC: int = 32
    zeroTol: float = 0
    nzQ: float = 1.0

    blockScale: str = "log"
    axNnzScale: str = "log"
    nzScale: str = "log"

    def Disabled():
        return HistParams(enabled=False)


@dataclass
class SingularParams:
    enabled: bool = True

    def Disabled():
        return SingularParams(enabled=False)


@dataclass
class CondParams:
    enabled: bool = True
    tol: float = 1e-4

    def Disabled():
        return CondParams(enabled=False)


@dataclass
class SpectrumParams:
    enabled: bool = True

    def Disabled():
        return SpectrumParams(enabled=False)


@dataclass
class SparsifyParams:
    enabled: bool = True
    minTol: float = 1e-30

    def Disabled():
        return SparsifyParams(enabled=False)


@dataclass
class BlockParams:
    enabled: bool = True
    tol: float = 0
    maxR: int = 16
    maxC: int = 16

    checkRC: list[tuple[int, int]] = dataclasses.field(
        default_factory=lambda: [(1, 1), (2, 1), (4, 1), (8, 1)]
    )

    cacheSize: int = 64
    floatSize: int = 8
    intSize: int = 4

    def Disabled():
        return SpectrumParams(enabled=False)


def SpringLayout(params, g, is3d):
    if GRAPH_ENABLED:
        return nx.spring_layout(g, seed=SEED, dim=(3 if is3d else 2))


@dataclass
class GraphParams:
    enabled: bool = True
    tol: float = -1
    layout = SpringLayout

    enableAnimation: bool = False
    animationFrames: int = 30
    animationDuration: float = 100

    def Disabled():
        return SpectrumParams(enabled=False)


@dataclass
class ReportParams:
    matrix: MatrixDescType | MatrixLoader
    name: str | None = None

    colorParams: ColorParams | NoneType = None
    shadeParams: ShadeParams | NoneType = None
    scatterParams: ScatterParams | NoneType = None

    histParams: HistParams | NoneType = None

    singularParams: SingularParams | NoneType = None
    condParams: CondParams | NoneType = None
    spectrumParams: SpectrumParams | NoneType = None

    sparsifyParams: SparsifyParams | NoneType = None

    blockParams: BlockParams | NoneType = None

    graphParams: GraphParams | NoneType = None


@jit
def ComputeSymmetry(
    m: int,
    n: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
):
    loCnt = 0
    upCnt = 0
    strCnt = 0
    numCnt = 0
    pairsCnt = 0

    for i in range(m):
        start = indptr[i]
        end = indptr[i + 1]

        for kk in range(end - start):
            k = start + kk
            j = indices[k]
            v = values[k]

            if i == j:
                continue

            jStart = indptr[j]
            jEnd = indptr[j + 1]
            kSymm = np.searchsorted(indices[jStart:jEnd], i) + jStart
            vSymm = np.nan
            if kSymm >= jStart and kSymm < jEnd and indices[kSymm] == i:
                vSymm = values[kSymm]

            if np.isnan(vSymm):
                pairsCnt += 1
                if j < i:
                    loCnt += 1
                else:
                    upCnt += 1
            else:
                if j < i:
                    pairsCnt += 1
                    strCnt += 1
                    if v == vSymm:
                        numCnt += 1

    loCnt = loCnt / pairsCnt * 100
    upCnt = upCnt / pairsCnt * 100
    strCnt = strCnt / pairsCnt * 100
    numCnt = numCnt / pairsCnt * 100

    return loCnt, strCnt, numCnt, upCnt


def FormatBytes(nbytes):
    log2 = np.log2(nbytes)

    if log2 <= 9:
        return nbytes, "b"
    elif log2 <= 19:
        return nbytes / (2**10), "Kb"
    elif log2 <= 29:
        return nbytes / (2**20), "Mb"
    else:
        return nbytes / (2**30), "Gb"


def ComputeImageSize(pxRows_, pxCols_, m, n):
    pxRows = pxRows_
    pxCols = pxCols_

    if pxRows <= 0 and pxCols <= 0:
        pxCols = DEFAULT_PX_COLS

    if pxRows <= 0:
        pxRows = int(m * pxCols / n)

    if pxCols <= 0:
        pxCols = int(n * pxRows / m)

    pxRows = min(pxRows, m)
    pxCols = min(pxCols, n)

    return pxRows, pxCols


@jit
def FormPxBlockLine(
    m: int,
    n: int,
    rowPtr: np.ndarray,
    col: np.ndarray,
    values: np.ndarray,
    rowPx: np.ndarray,
    rowPxSize: np.ndarray,
    iStart: int,
    iEnd: int,
    pxCols: int,
):
    for p in range(iEnd - iStart):
        i = iStart + p
        start = rowPtr[i]
        end = rowPtr[i + 1]
        for kk in range(end - start):
            k = start + kk
            v = values[k]
            j = col[k]
            jb = j * pxCols // n
            rowPx[jb, rowPxSize[jb]] = v
            rowPxSize[jb] += 1


def ProcessPixelBlocks(pxRows, pxCols, matrix, process):
    m, n = matrix.shape

    pxSize = int(math.ceil(m / pxRows)) * int(math.ceil(n / pxCols))

    for ib in range(pxRows):
        rowPx = np.zeros((pxCols, pxSize))
        rowPxSize = np.zeros(pxCols, dtype=int)

        iStart = int(math.ceil(ib * m / pxRows))
        iEnd = min(int(math.ceil((ib + 1) * m / pxRows)), m)

        FormPxBlockLine(
            m,
            n,
            matrix.indptr,
            matrix.indices,
            matrix.data,
            rowPx,
            rowPxSize,
            iStart,
            iEnd,
            pxCols,
        )

        for jb in range(pxCols):
            r = iEnd - iStart
            jStart = int(math.ceil(jb * n / pxCols))
            jEnd = min(int(math.ceil((jb + 1) * n / pxCols)), n)
            c = jEnd - jStart
            process(ib, jb, r, c, rowPx[jb, : rowPxSize[jb]])


def SaveImage(path, data):
    m = data.shape[0]
    n = data.shape[1]
    t = int(np.ceil(MIN_PNG_SIZE / min(m, n)))

    multMat = np.zeros((t, t, 1))
    for i in range(t):
        for j in range(t):
            multMat[i, j, 0] = 1

    imageData = np.kron(data, multMat)

    plt.imsave(path, imageData)


def SaveFig(path, fig):
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")


@jit
def CountRowColNnz(
    m: int,
    n: int,
    rowPtr: np.ndarray,
    col: np.ndarray,
    values: np.ndarray,
    tol: float,
):
    rowNnz = np.zeros(m, dtype=np.int64)
    colNnz = np.zeros(n, dtype=np.int64)

    for i in range(m):
        start = rowPtr[i]
        end = rowPtr[i + 1]

        for kk in range(end - start):
            k = start + kk
            j = col[k]
            v = values[k]

            if abs(v) <= tol:
                continue

            rowNnz[i] += 1
            colNnz[j] += 1

    return rowNnz, colNnz


@jit
def CountKrc(
    m: int,
    n: int,
    rowPtr: np.ndarray,
    col: np.ndarray,
    values: np.ndarray,
    tol: float,
    rList: list[int],
    cList: list[int],
):
    kr = len(rList)
    kc = len(cList)

    krc = np.zeros((kr, kc))

    counts = np.zeros((kr, kc, n), dtype=np.int64)
    lists = np.zeros((kr, kc, n), dtype=np.int64)
    sizes = np.zeros((kr, kc), dtype=np.int64)

    for i in range(m):
        start = rowPtr[i]
        end = rowPtr[i + 1]

        for ir in range(kr):
            r = rList[ir]
            if i % r == 0:
                for ic in range(kc):
                    for k in range(sizes[ir, ic]):
                        counts[ir, ic, lists[ir, ic, k]] = 0
                    sizes[ir, ic] = 0

        for kk in range(end - start):
            k = start + kk
            j = col[k]
            v = values[k]

            if abs(v) <= tol:
                continue

            for ir in range(kr):
                for ic in range(kc):
                    jb = j // cList[ic]

                    if counts[ir, ic, jb] == 0:
                        lists[ir, ic, sizes[ir, ic]] = jb
                        krc[ir, ic] += 1
                        sizes[ir, ic] += 1

                    counts[ir, ic, jb] += 1

    return krc, counts


def ViewMatrix(rotation: float):
    PLANE_ANGLE = 0.3

    ct = np.cos(PLANE_ANGLE)
    st = np.sin(PLANE_ANGLE)

    cr = np.cos(rotation)
    sr = np.sin(rotation)

    proj = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -st, ct],
        ],
    )
    rot = np.array(
        [
            [cr, sr, 0.0],
            [-sr, cr, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    view = proj @ rot

    return view


# posArray should be normalized to [0,1]
def CreateGraphImage(graph, floatPosArray):
    POINT_RADIUS = 3
    LINE_WIDTH = 1.3
    IMAGE_SIZE = 1024

    def PointCmap(t):
        r, g, b, a = matplotlib.colormaps["jet"](t)
        return int(r * 255), int(g * 255), int(b * 255)

    n = floatPosArray.shape[1]

    imPosArray = (floatPosArray * (IMAGE_SIZE - 1)).astype(np.int16)

    im = PIL.Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
    draw = aggdraw.Draw(im)

    for i1, i2, rest in nx.to_edgelist(graph):
        i1 = int(i1)
        i2 = int(i2)

        if i1 == i2:
            continue

        color = (0, 0, 0)

        pen = aggdraw.Pen(color, LINE_WIDTH)
        draw.line(
            [
                imPosArray[0, i1],
                imPosArray[1, i1],
                imPosArray[0, i2],
                imPosArray[1, i2],
            ],
            pen,
        )

    for i in range(n):
        color = PointCmap(i / (n - 1))

        x0 = imPosArray[0, i]
        y0 = imPosArray[1, i]

        x0, x1 = x0 - POINT_RADIUS, x0 + POINT_RADIUS
        y0, y1 = y0 - POINT_RADIUS, y0 + POINT_RADIUS

        brush = aggdraw.Brush(color)
        draw.ellipse([x0, y0, x1, y1], brush, None)

    draw.flush()
    return im


@dataclass
class DataText:
    heading: str | None
    text: str
    figN: str | None = None

    def FormFigureText(self):
        return f"<td>{self.text}<br/>Figure {self.figN}. {self.heading}</td>"

    def FormDataText(self):
        headtext = ""
        if self.heading is not None:
            headtext = f"<u>{self.heading}</u>"

        return f"<tt>{headtext}<br/>{self.text}</tt>"


class HTMLOutput:
    name: str
    dataTexts: list[DataText]
    figures: list[DataText]

    def __init__(self, name):
        self.name = name
        self.dataTexts = []
        self.figures = []

    def AddFigure(self, heading, html):
        figN = len(self.figures) + 1
        self.figures.append(DataText(heading=heading, figN=figN, text=html))
        return figN

    def AddData(self, heading, text):
        self.dataTexts.append(DataText(heading=heading, text=text))

    def FormLine(self):
        figRows = "<tr>"
        for i in range(len(self.figures)):
            fig = self.figures[i]

            if i > 0 and i % COL_PER_ROW_MAX == 0:
                figRows += "</tr><tr>"
            figRows += fig.FormFigureText()
        figRows += "</tr>"

        dataCells = []
        dataSizes = []
        for d in self.dataTexts:
            text = d.FormDataText()
            dataCells.append(text)
            dataSizes.append(text.count("<br/>"))

        textLines = sum(dataSizes) + len(dataSizes) - 1

        colLines = 0
        dataCols = [""]

        for i in range(len(dataCells)):
            if colLines > textLines / COL_PER_ROW_MAX:
                dataCols.append("")
                colLines = 0

            if colLines != 0:
                dataCols[-1] += "<br/>"

            dataCols[-1] += dataCells[i]
            colLines += dataSizes[i]

        dataRow = "".join(f"<td>{t}</td>" for t in dataCols)
        dataRow = f"<tr>{dataRow}</tr>"

        return (
            f'<tr class="page-break"><td><tt><b>{self.name}</b></tt></td></tr>'
            + dataRow
            + figRows
        )


class VerboseLogger:
    startTime: float
    sectionTime: float
    sectionName: float
    verbose: bool

    def __init__(self):
        self.startTime = time.perf_counter()

    def StartSection(self, name):
        self.sectionTime = time.perf_counter()
        self.sectionName = name

    def FinishSection(self):
        now = time.perf_counter()
        ela = now - self.sectionTime
        print(f"-> {self.sectionName:20s} ({ela:6.2f} s)")

    def Finish(self):
        now = time.perf_counter()
        ela = now - self.startTime
        print(f"{'Finished all':23s} ({ela:6.2f} s)")


class NoLogger:
    def __init__(self):
        pass

    def StartSection(self, _):
        pass

    def FinishSection(self):
        pass

    def Finish(self):
        pass


def CreateLine(
    matrix: CSRMatrix,
    params: ReportParams,
    outDir: str,
    verbose: bool,
):
    logger = NoLogger()

    if verbose:
        logger = VerboseLogger()

    # --- Basic info ---

    logger.StartSection("Basic info")

    m, n = matrix.shape
    nnz = matrix.getnnz()
    sparsity = nnz / (n * m) * 100

    imagePrefix = os.path.join(IMAGE_DIR, params.name)

    posArr = matrix.data[matrix.data > 0]
    negArr = matrix.data[matrix.data < 0]
    minPos, maxPos, minNeg, maxNeg = math.nan, math.nan, math.nan, math.nan
    if len(posArr) > 0:
        minPos = posArr.min()
        maxPos = posArr.max()
    if len(negArr) > 0:
        minNeg = negArr.min()
        maxNeg = negArr.max()

    matrixDense = None

    bandLower, bandUpper = scipy.sparse.linalg.spbandwidth(matrix)

    lo, struc, numeric, up = ComputeSymmetry(
        m, n, matrix.indptr, matrix.indices, matrix.data
    )

    csrSize, csrSizeUnits = FormatBytes(nnz * (8 + 4) + m * 4)
    denseSize, denseSizeUnits = FormatBytes(n * m * 8)

    out = HTMLOutput(params.name)

    out.AddData(
        "General data",
        f"{m} x {n}, nnz = {nnz} ({sparsity:8.4f}%)<br/>"
        + f"pos range = ({minPos:11.4e}, {maxPos:11.4e})<br/>"
        + f"neg range = ({minNeg:11.4e}, {maxNeg:11.4e})<br/>"
        + f"bandwidth = ({bandLower:11d}, {bandUpper:11d})<br/>"
        + f"symmetry =<br/>{lo:8.4f}% / {struc:8.4f}% ({numeric:8.4f}%) / {up:8.4f}%<br/>"
        + f"size as CSR   = {csrSize:8.4f} {csrSizeUnits}<br/>"
        + f"size as Dense = {denseSize:8.4f} {denseSizeUnits}<br/>",
    )

    logger.FinishSection()

    # --- Color image ---

    if params.colorParams.enabled:
        logger.StartSection("Color image")

        filepath = imagePrefix + "-px.png"
        pxRows, pxCols = ComputeImageSize(
            params.colorParams.pxRows, params.colorParams.pxCols, m, n
        )
        imageData = np.zeros((pxRows, pxCols, 3))

        params.colorParams.colormap.SetMax(maxPos, minNeg)

        def Process(ib, jb, r, c, values):
            color = GetValuesColor(list(values), params.colorParams.colormap)
            imageData[ib, jb] = color

        ProcessPixelBlocks(pxRows, pxCols, matrix, Process)

        SaveImage(os.path.join(outDir, filepath), imageData)

        out.AddFigure(
            "Color image",
            f'<img class="pixelated" src="{filepath}" />',
        )

        logger.FinishSection()

    # --- Shade image ---

    if params.shadeParams.enabled:
        logger.StartSection("Shade image")

        filepath = imagePrefix + "-shade.png"
        pxRows, pxCols = ComputeImageSize(
            params.colorParams.pxRows, params.colorParams.pxCols, m, n
        )
        densityData = np.zeros((pxRows, pxCols))
        imageData = np.zeros((pxRows, pxCols, 3))

        maxPxSize = np.zeros(1, dtype=int)
        minPxSize = np.zeros(1, dtype=int)
        minPxSize[0] = INT_MAX
        maxFill = np.zeros(1, dtype=int)

        def Process(ib, jb, r, c, values):
            nzvalues = [v for v in values if abs(v) > params.shadeParams.zeroTol]
            nnz = len(nzvalues)
            densityData[ib, jb] = len(nzvalues) / (r * c)
            maxPxSize[0] = max(r * c, maxPxSize[0])
            minPxSize[0] = min(r * c, maxPxSize[0])
            maxFill[0] = max(nnz, maxFill[0])

        ProcessPixelBlocks(pxRows, pxCols, matrix, Process)

        maxDensity = densityData.max()
        white = np.array([1, 1, 1])
        black = np.array([0, 0, 0])
        red = np.array([1, 0, 0])

        for ib in range(pxRows):
            for jb in range(pxCols):
                z = densityData[ib, jb] / maxDensity
                color = None
                if z == 0:
                    color = white
                elif z < 0.5:
                    color = white + (red - white) * (z + 0.1)
                else:
                    color = white + (black - white) * z
                imageData[ib, jb] = color

        SaveImage(os.path.join(outDir, filepath), imageData)

        figN = out.AddFigure(
            "Shade image", f'<img class="pixelated" src="{filepath}" />'
        )

        out.AddData(
            f"Shade data (Figure {figN})",
            ""
            + "max px fill =<br/>"
            + f"{maxFill[0]:5d}/{minPxSize[0]:5d}-{maxPxSize[0]:5d} "
            + f"({maxFill[0] / maxPxSize[0] * 100:8.4f}%)<br/>",
        )

        logger.FinishSection()

    # --- Scatter image ---

    if params.scatterParams.enabled:
        logger.StartSection("Scatter image")

        filepath = imagePrefix + "-scatter.png"

        params.scatterParams.colormap.SetMax(maxPos, minNeg)

        fig, ax = plt.subplots(figsize=(8, 8))

        rows = np.zeros(nnz)
        cols = np.zeros(nnz)
        colors = np.zeros((nnz, 4))

        for i in range(m):
            start = matrix.indptr[i]
            end = matrix.indptr[i + 1]
            for kk in range(end - start):
                k = start + kk
                v = matrix.data[k]
                j = matrix.indices[k]
                color = GetValuesColor([v], params.scatterParams.colormap)

                rows[k] = i
                cols[k] = j
                colors[k, :3] = color
                colors[k, 3] = params.scatterParams.alpha

        ax.scatter(cols, rows, params.scatterParams.pointSize, color=colors)

        ax.grid()
        ax.set(
            xlim=[-PLOT_MARGIN * (n - 1), (1 + PLOT_MARGIN) * (n - 1)],
            xlabel="Cols",
            ylim=[-PLOT_MARGIN * (m - 1), (1 + PLOT_MARGIN) * (m - 1)],
            ylabel="Rows",
        )

        ax.invert_yaxis()
        fig.tight_layout()
        SaveFig(os.path.join(outDir, filepath), fig)
        plt.close(fig)

        out.AddFigure("Scatter nonzeros", f'<img src="{filepath}" />')

        logger.FinishSection()

    # --- Histogram ---

    if params.histParams.enabled:
        logger.StartSection("Histogram")

        filepath = imagePrefix + "-hist.png"

        r, c = params.histParams.blockR, params.histParams.blockC
        blocksNnz = np.zeros(r * c, dtype=np.int64)
        maxBNnz = 0

        for ib in range((m + r - 1) // r):
            i = ib * r
            mb = min(i + r, m) - i
            (_, counts) = CountKrc(
                mb,
                n,
                matrix.indptr[i : i + mb + 1],
                matrix.indices,
                matrix.data,
                params.histParams.zeroTol,
                [r],
                [c],
            )

            for jb in range((n + c - 1) // c):
                bnnz = counts[0, 0, jb]
                if bnnz != 0:
                    blocksNnz[bnnz - 1] += 1
                    maxBNnz = max(maxBNnz, bnnz)

        blocksNnz = blocksNnz[:maxBNnz]

        rowNnz, colNnz = CountRowColNnz(
            m,
            n,
            matrix.indptr,
            matrix.indices,
            matrix.data,
            params.histParams.zeroTol,
        )
        maxRowNnz = rowNnz.max()
        maxColNnz = colNnz.max()
        maxAxNnz = max(maxRowNnz, maxColNnz)

        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        ax1, ax2, ax3 = axs.flatten()

        ax1.set(
            title=f"{r} x {c} blocks sparsity (max={maxBNnz}/{r * c}, {maxBNnz / (r * c) * 100:.4f}%)",
            xlabel="Sparsity, %",
            ylabel="Count",
            yscale=params.histParams.blockScale,
        )
        nbins = min(maxBNnz, MAX_HIST_BINS)
        bins = np.linspace(0, maxBNnz / (r * c) * 100, nbins + 1)
        ax1.hist(
            (np.arange(maxBNnz) + 1) / (r * c) * 100,
            bins,
            weights=blocksNnz,
            color="k",
            edgecolor="r",
        )

        ax2.set(
            title="Row and col nnz "
            + f"(row max={maxRowNnz}/{n}, {maxRowNnz / n * 100:.4f}%,"
            + f" col max={maxColNnz}/{m}, {maxColNnz / m * 100:.4f}%)",
            xlabel="Nnz",
            ylabel="Count",
            yscale=params.histParams.axNnzScale,
        )
        joinFactor = int(np.ceil(maxAxNnz / (MAX_HIST_BINS // 2)))
        nbins = (maxAxNnz + joinFactor - 1) // joinFactor
        bins = np.arange(nbins + 1) * joinFactor + 0.5
        ax2.hist(
            np.array([rowNnz, colNnz]).T,
            bins,
            rwidth=1,
            color=["k", "r"],
            label=["row", "col"],
        )
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax2.legend()

        ax3.set(
            title=f"Entry values ({params.histParams.nzQ * 100}%)",
            xlabel="Value",
            ylabel="Count",
            yscale=params.histParams.nzScale,
        )
        nzData = matrix.data[:]
        minV = nzData.min()
        maxV = nzData.max()
        if params.histParams.nzQ != 1.0:
            nzData.sort()
            s = (1 - params.histParams.nzQ) / 2
            i0 = int(np.floor(nnz * s))
            i1 = int(np.ceil(nnz * (1 - s)))
            nzData = nzData[i0:i1]
            minV = nzData[0]
            maxV = nzData[-1]
        nbins = MAX_HIST_BINS
        bins = np.linspace(minV, maxV, nbins + 1)
        ax3.hist(nzData, bins, color="k", edgecolor="r")

        fig.tight_layout()
        SaveFig(os.path.join(outDir, filepath), fig)
        plt.close(fig)

        out.AddFigure("Histograms", f'<img src="{filepath}" />')

        logger.FinishSection()

    # --- Singular values ---

    if params.singularParams.enabled:
        logger.StartSection("Singular values")

        filepath = imagePrefix + "-sing.png"

        if matrixDense is None:
            matrixDense = matrix.todense()

        singularValues = np.linalg.svd(
            matrixDense, full_matrices=False, compute_uv=False
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(list(range(len(singularValues))), singularValues, "k-")

        ax.grid()
        ax.set(yscale="log")

        fig.tight_layout()
        SaveFig(os.path.join(outDir, filepath), fig)
        plt.close(fig)

        singMin = singularValues[-1]
        singMax = singularValues[0]

        cond2 = 0

        if abs(singMin) < 1e-16:
            cond2 = +np.inf
        else:
            cond2 = singMax / singMin

        figN = out.AddFigure("Singular values", f'<img src="{filepath}" />')

        out.AddData(
            f"Singular values (Figure {figN})",
            ""
            + f"s range = ({singMin:11.4e}, {singMax:11.4e})<br/>"
            + f"k(p=2)  = {cond2:11.4e}<br/>",
        )

        logger.FinishSection()

    # --- Condition number ---

    elif params.condParams.enabled:
        logger.StartSection("Condition number")

        tol = params.condParams.tol

        singMin, singMax = np.nan, np.nan

        if m * n * 8 <= 2**30:  # Limit dense matrix to 1Gb
            if matrixDense is None:
                matrixDense = matrix.todense()
            singMin = np.linalg.norm(matrixDense, ord=-2)
            singMax = np.linalg.norm(matrixDense, ord=2)
        else:
            try:
                singMax = scipy.sparse.linalg.svds(
                    matrix,
                    k=1,
                    which="LM",
                    tol=tol,
                    random_state=SEED,
                    return_singular_vectors=False,
                )
                singMax = singMax[0]

                singMin = scipy.sparse.linalg.svds(
                    matrix,
                    k=1,
                    which="SM",
                    tol=tol,
                    random_state=SEED,
                    return_singular_vectors=False,
                )
                singMin = singMin[0]
            except Exception as e:
                print(
                    "WARN: did not compute condition number because "
                    "of the following exception:"
                )
                print(e)
                singMin, singMax = np.nan, np.nan

        cond2 = None
        if abs(singMin) < 1e-16:
            cond2 = +np.inf
        else:
            cond2 = singMax / singMin

        cond2Max = None
        if singMin < tol + 1e-16:
            cond2Max = +np.inf
        else:
            cond2Max = (singMax + tol) / (singMin - tol)

        cond2Min = (singMax - tol) / (singMin + tol)

        if np.isnan(singMin):
            out.AddData("Approximate conditioning", "Did not compute")
        else:
            out.AddData(
                "Approximate conditioning",
                ""
                + f"s range = ({singMin:11.4e}, {singMax:11.4e})<br/>"
                + f"          ({singMin - tol:11.4e}, {singMax + tol:11.4e})<br/>"
                + f"k(p=2)  = {cond2:11.4e} ({cond2Min:11.4e}, {cond2Max:11.4e})<br/>",
            )

        logger.FinishSection()

    # --- Spectrum ---

    if params.spectrumParams.enabled:
        logger.StartSection("Spectrum")

        filepath = imagePrefix + "-spectrum.png"

        if matrixDense is None:
            matrixDense = matrix.todense()

        spectrum = scipy.linalg.eigvals(matrixDense)

        plotData = np.zeros((len(spectrum), 3))
        uniqueValues = 0
        realCount = 0

        for z in spectrum:
            if abs(z.imag) < 1e-16:
                realCount += 1
            multiple = False
            for k in range(uniqueValues):
                if abs(z - (plotData[k, 0] + plotData[k, 1] * 1j)) < 1e-16:
                    plotData[k, 2] += 1
                    multiple = True
                break
            if not multiple:
                plotData[uniqueValues] = (z.real, z.imag, 1)
                uniqueValues += 1

        plotData = plotData[:uniqueValues]
        maxMult = plotData[:uniqueValues, 2].max()

        colors = np.zeros((uniqueValues, 4))

        for k in range(uniqueValues):
            z = (plotData[k, 2] - 1) / maxMult
            colors[k] = (z, 0, 1 - z, 0.5)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(plotData[:, 0], plotData[:, 1], color=colors)

        ax.grid()
        ax.set(xlabel="Real", ylabel="Imaginary")

        fig.tight_layout()
        SaveFig(os.path.join(outDir, filepath), fig)
        plt.close(fig)

        absArr = np.zeros(uniqueValues)

        for k in range(uniqueValues):
            absArr[k] = np.sqrt(plotData[k, 0] ** 2 + plotData[k, 1] ** 2)

        figN = out.AddFigure("Spectrum", f'<img src="{filepath}" />')

        out.AddData(
            f"Spectrum (Figure {figN})",
            ""
            + f"real range = ({plotData[:, 0].min():11.4e}, {plotData[:, 0].max():11.4e})<br/>"
            + f"imag range = ({plotData[:, 1].min():11.4e}, {plotData[:, 1].max():11.4e})<br/>"
            + f"abs  range = ({absArr.min():11.4e}, {absArr.max():11.4e})<br/>"
            + f"real l     = {realCount} ({realCount / n * 100:8.4f}%)<br/>"
            + f"max mult   = {int(maxMult)}<br/>",
        )

        logger.FinishSection()

    # --- Sparsify ---

    if params.sparsifyParams.enabled:
        logger.StartSection("Sparsify")

        filepath = imagePrefix + "-sparsify.png"

        nzData = np.abs(matrix.data)
        nzData.sort()
        minTol = params.sparsifyParams.minTol
        nZero = 0
        while nzData[nZero] < minTol and nZero < len(nzData):
            nZero += 1
        nzData = nzData[nZero:]

        dnnz = len(nzData)
        tols = np.zeros(2 * (dnnz + 1))
        sparsity = np.zeros(2 * (dnnz + 1))
        frobNorm = np.zeros(2 * (dnnz + 1))

        tols[0] = nzData[0] / 2
        tols[1 : 2 * dnnz + 1 : 2] = nzData
        tols[2 : 2 * dnnz + 1 : 2] = tols[1 : 2 * dnnz + 1 : 2]
        tols[-1] = nzData[-1] * 2

        sparsity[0::2] = (dnnz - np.arange(dnnz + 1)) / (m * n) * 100
        sparsity[1::2] = sparsity[::2]

        frobNorm[0] = 0
        frobNorm[2::2] = np.sqrt(np.cumsum(np.square(nzData)))
        frobNorm[1::2] = frobNorm[::2]

        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()

        ax1.set(
            xlabel="Tolerance",
            xscale="log",
            ylabel="Frobenius norm",
            yscale="log",
        )
        ax2.set(ylabel="Sparsity, %")

        p1 = ax1.plot(tols, frobNorm, "k-", label="Frobenius norm")
        p2 = ax2.plot(tols, sparsity, "r-", label="Sparsity")

        lns = p1 + p2
        ax1.legend(lns, [e.get_label() for e in lns])

        fig.tight_layout()
        SaveFig(os.path.join(outDir, filepath), fig)
        plt.close(fig)

        out.AddFigure("Sparsify", f'<img src="{filepath}" />')

        logger.FinishSection()

    # --- Block ---

    if params.blockParams.enabled:
        logger.StartSection("Block")

        filepath = imagePrefix + "-block.png"

        maxR = params.blockParams.maxR
        maxC = params.blockParams.maxC
        (krc, _) = CountKrc(
            m,
            n,
            matrix.indptr,
            matrix.indices,
            matrix.data,
            params.blockParams.tol,
            [k + 1 for k in range(maxR)],
            [k + 1 for k in range(maxC)],
        )

        misses = np.zeros((maxR, maxC))
        sc = params.blockParams.cacheSize
        sf = params.blockParams.floatSize
        si = params.blockParams.intSize

        for rm in range(maxR):
            for cm in range(maxC):
                r = rm + 1
                c = cm + 1
                bnnz = krc[rm, cm]

                misses[rm, cm] = (
                    bnnz * (r * c * sf + si)
                    + ((m + r - 1) // r + 1) * si
                    + m * sf
                    + n * sf
                ) / sc

        idx = np.argmin(misses)
        rm, cm = np.unravel_index(idx, misses.shape)
        minR = rm + 1
        minC = cm + 1
        minMiss = misses[rm, cm]

        improve = misses / minMiss - 1

        fig, ax = plt.subplots(figsize=(8, 8))

        ims = ax.imshow(
            improve,
            cmap="magma",
            extent=(0.5, maxR + 0.5, 0.5, maxC + 0.5),
            origin="lower",
        )

        ax.set(xlabel="Block cols", ylabel="Block rows")
        ax.invert_yaxis()

        fig.colorbar(ims, shrink=0.7)
        fig.tight_layout()

        SaveFig(os.path.join(outDir, filepath), fig)
        plt.close(fig)

        figN = out.AddFigure("Blocks", f'<img src="{filepath}" />')

        checkLines = ""

        for r, c in params.blockParams.checkRC:
            bnd = misses[r - 1, c - 1]
            imp = improve[r - 1, c - 1]
            checkLines += f"{r:2d} x {c:2d}: {bnd:11.4e} (+{imp * 100:8.4f}%)<br/>"

        out.AddData(
            f"Blocks (Figure {figN})",
            ""
            + f"Min lower bound at {minR} x {minC}<br/>"
            + f"          with >{minMiss:11.4e} misses<br/>"
            + checkLines,
        )

        logger.FinishSection()

    # --- Graph ---

    if params.graphParams.enabled:
        logger.StartSection("Graph")

        filepath = imagePrefix + "-graph.png"

        g = nx.Graph()
        g.add_nodes_from(range(n))

        edgelist = np.zeros((nnz, 2), dtype=np.int64)
        edgelistSize = 0
        for i in range(m):
            start = matrix.indptr[i]
            end = matrix.indptr[i + 1]
            for kk in range(end - start):
                k = start + kk
                j = matrix.indices[k]
                v = matrix.data[k]
                if abs(v) > params.graphParams.tol and i != j:
                    edgelist[edgelistSize] = (int(i), int(j))
                    edgelistSize += 1
        edgelist = edgelist[:edgelistSize]
        g.add_edges_from(edgelist)

        layout = params.graphParams.layout
        posDict = layout(g, params.graphParams.enableAnimation)
        pos = np.zeros((3 if params.graphParams.enableAnimation else 2, n))
        for i in range(n):
            pos[:, i] = posDict[i]

        # 2d image

        mat = np.eye(2)
        if params.graphParams.enableAnimation:
            mat = ViewMatrix(0)
        posArray = mat @ pos

        minX = posArray[0, :].min()
        maxX = posArray[0, :].max()
        minY = posArray[1, :].min()
        maxY = posArray[1, :].max()

        minX, maxX = (
            minX - PLOT_MARGIN * (maxX - minX),
            maxX + PLOT_MARGIN * (maxX - minX),
        )
        minY, maxY = (
            minY - PLOT_MARGIN * (maxY - minY),
            maxY + PLOT_MARGIN * (maxY - minY),
        )

        posArray[0, :] = (posArray[0, :] - minX) / (maxX - minX)
        posArray[1, :] = (posArray[1, :] - minY) / (maxY - minY)

        im = CreateGraphImage(g, posArray)
        im.save(os.path.join(outDir, filepath), "PNG")

        out.AddFigure("Graph", f'<img src="{filepath}" />')

        logger.FinishSection()

        # 3d gif

        if params.graphParams.enableAnimation:
            logger.StartSection("Graph 3d")

            filepath = imagePrefix + "-graph3d.gif"

            nFrames = params.graphParams.animationFrames
            posArrays = np.zeros((nFrames, 2, n))
            for f in range(nFrames):
                #  t = 2 * np.pi / nFrames * f
                #  angle = (np.pi / 6) * np.sin(t)
                #  angle = t

                t = 2 * f / nFrames
                angle = (np.pi / 6) * np.abs(1 - t)

                posArrays[f] = ViewMatrix(angle) @ pos

            minX = posArrays[:, 0, :].min()
            maxX = posArrays[:, 0, :].max()
            minY = posArrays[:, 1, :].min()
            maxY = posArrays[:, 1, :].max()

            minX, maxX = (
                minX - PLOT_MARGIN * (maxX - minX),
                maxX + PLOT_MARGIN * (maxX - minX),
            )
            minY, maxY = (
                minY - PLOT_MARGIN * (maxY - minY),
                maxY + PLOT_MARGIN * (maxY - minY),
            )

            posArrays[:, 0, :] = (posArrays[:, 0, :] - minX) / (maxX - minX)
            posArrays[:, 1, :] = (posArrays[:, 1, :] - minY) / (maxY - minY)

            images = None

            with multiprocessing.Pool() as p:
                images = p.starmap(
                    CreateGraphImage, [(g, posArrays[f]) for f in range(nFrames)]
                )

            images[0].save(
                os.path.join(outDir, filepath),
                "GIF",
                save_all=True,
                append_images=images[1:],
                duration=params.graphParams.animationDuration,
                loop=0,
            )

            out.AddFigure("Animated 3D graph", f'<img src="{filepath}" />')

            logger.FinishSection()

    # --- Tail ---

    line = out.FormLine()
    logger.Finish()

    return line


# ----------------------
# --- Default choice ---

INT_MAX = sys.maxsize


@dataclass
class MatrixLimit:
    maxM: int = INT_MAX
    maxN: int = INT_MAX
    maxNNZ: int = INT_MAX

    def Square(n: int = INT_MAX, nnz: int = INT_MAX):
        return MatrixLimit(maxM=n, maxN=n, maxNNZ=nnz)

    def Disabled():
        return MatrixLimit(maxNNZ=-1)

    def IsWithin(self, matrix: CSRMatrix):
        n, m = matrix.shape
        nnz = matrix.getnnz()

        return n <= self.maxN and m <= self.maxM and nnz <= self.maxNNZ


@dataclass
class DefaultChoiceParams:
    defaultColorParams: ColorParams = ColorParams()
    limitColor: MatrixLimit = MatrixLimit()

    defaultShadeParams: ShadeParams = ShadeParams()
    limitShade: MatrixLimit = MatrixLimit()

    defaultScatterParams: ScatterParams = ScatterParams()
    limitScatter: MatrixLimit = MatrixLimit(maxNNZ=100_000)

    defaultHistParams: HistParams = HistParams()
    limitHist: MatrixLimit = MatrixLimit()

    defaultSingularParams: SingularParams = SingularParams()
    limitSingular: MatrixLimit = MatrixLimit.Square(n=1_000)

    defaultCondParams: CondParams = CondParams()
    limitCond: MatrixLimit = MatrixLimit(maxNNZ=10_000)

    defaultSpectrumParams: SpectrumParams = SpectrumParams()
    limitSpectrum: MatrixLimit = MatrixLimit.Square(n=1_000)

    defaultSparsifyParams: SparsifyParams = SparsifyParams()
    limitSparsify: MatrixLimit = MatrixLimit()

    defaultBlockParams: BlockParams = BlockParams()
    limitBlock: MatrixLimit = MatrixLimit()

    defaultGraphParams: GraphParams = GraphParams()
    limitGraph: MatrixLimit = MatrixLimit.Square(n=1_000, nnz=10_000)


BLOCKS_PRESET = DefaultChoiceParams(
    limitScatter=MatrixLimit.Disabled(),
    limitSingular=MatrixLimit.Disabled(),
    limitCond=MatrixLimit.Disabled(),
    limitSpectrum=MatrixLimit.Disabled(),
    limitGraph=MatrixLimit.Disabled(),
)


def LoadMatrix(loader: MatrixDescType | MatrixLoader):
    if callable(loader):
        return LoadMatrix(loader())

    if type(loader) is NamedMatrixPath:
        path = loader.path
        name = loader.name

        mat = None

        if path.endswith(".mtx"):
            mat = scipy.io.mmread(path).tocsr()

        if path.endswith(".npz"):
            data = np.load(path)
            m = int(data["nrows"][0])
            n = int(data["ncols"][0])
            mat = CSRMatrix(
                (data["values"], data["col_indices"], data["row_ptr"]),
                shape=(m, n),
            )

        return LoadMatrix(NamedCSRMatrix(matrix=mat, name=name))

    if type(loader) is str:
        return LoadMatrix(NamedMatrixPath(path=loader, name=loader))

    if type(loader) is CSRMatrix:
        return LoadMatrix(NamedCSRMatrix(matrix=loader, name=""))

    if type(loader) is NamedCSRMatrix:
        return loader

    return None


def FillDefaultParams(
    params: ReportParams,
    matrix: CSRMatrix,
    retName: str | None,
    i: int,
    defaultChoice: DefaultChoiceParams,
):
    m, n = matrix.shape

    if params.name is None:
        if retName == "":
            params.name = f"Matrix {i}"
        else:
            params.name = retName

    def EnableIf(param, default, limit):
        if param is None:
            if limit.IsWithin(matrix):
                return default
            else:
                return type(default).Disabled()
        else:
            return param

    params.colorParams = EnableIf(
        params.colorParams,
        defaultChoice.defaultColorParams,
        defaultChoice.limitColor,
    )

    params.shadeParams = EnableIf(
        params.shadeParams,
        defaultChoice.defaultShadeParams,
        defaultChoice.limitShade,
    )

    params.scatterParams = EnableIf(
        params.scatterParams,
        defaultChoice.defaultScatterParams,
        defaultChoice.limitScatter,
    )

    params.histParams = EnableIf(
        params.histParams,
        defaultChoice.defaultHistParams,
        defaultChoice.limitHist,
    )

    params.singularParams = EnableIf(
        params.singularParams,
        defaultChoice.defaultSingularParams,
        defaultChoice.limitSingular,
    )

    params.condParams = EnableIf(
        params.condParams,
        defaultChoice.defaultCondParams,
        defaultChoice.limitCond,
    )

    if params.spectrumParams is None:
        if m == n:
            params.spectrumParams = EnableIf(
                params.spectrumParams,
                defaultChoice.defaultSpectrumParams,
                defaultChoice.limitSpectrum,
            )
        else:
            params.spectrumParams = SpectrumParams.Disabled()

    if params.spectrumParams.enabled and m != n:
        print(
            f"WARN: will not compute spectrum of {params.name}, "
            + "since it is not square"
        )
        params.spectrumParams = SpectrumParams.Disabled()

    params.sparsifyParams = EnableIf(
        params.sparsifyParams,
        defaultChoice.defaultSparsifyParams,
        defaultChoice.limitSparsify,
    )

    params.blockParams = EnableIf(
        params.blockParams,
        defaultChoice.defaultBlockParams,
        defaultChoice.limitBlock,
    )

    if params.blockParams.enabled and (2 ** (8 * params.blockParams.intSize)) < max(
        m, n
    ):
        print(
            f"WARN: intSize={params.blockParams.intSize} bytes could not "
            + f"fit indices of a {m} x {n} matrix"
        )

        if AUTO_INCREASE_INT_SIZE:
            while (2 ** (8 * params.blockParams.intSize)) < max(m, n):
                params.blockParams.intSize *= 2
            print(f"WARN: set intSize={params.blockParams.intSize} bytes")

    if params.graphParams is None:
        if m == n:
            params.graphParams = EnableIf(
                params.graphParams,
                defaultChoice.defaultGraphParams,
                defaultChoice.limitGraph,
            )
        else:
            params.graphParams = GraphParams.Disabled()

    if params.graphParams.enabled and not GRAPH_ENABLED:
        print(
            f"WARN: will not compute graph layout of {params.name}, "
            + "since networkx is not installed"
        )
        params.graphParams = GraphParams.Disabled()

    if params.graphParams.enabled and m != n:
        print(
            f"WARN: will not compute graph layout of {params.name}, "
            + "since it is not square"
        )
        params.graphParams = GraphParams.Disabled()

    return params


# -------------------
# --- Entrypoints ---


def CreateReport(
    matrices: list[ReportParams],
    outDir: str,
    defaultChoiceParams: DefaultChoiceParams = DefaultChoiceParams(),
    verbose: bool = False,
):
    os.makedirs(os.path.join(outDir, IMAGE_DIR), exist_ok=True)

    print(f"Output dir: {outDir}, index file: {os.path.join(outDir, REPORT_NAME)}")

    output = (
        "<html><style>"
        + "img { width: 500; "
        + "border: 1px solid black; }\n"
        + ".pixelated { image-rendering: pixelated; "
        + "image-rendering: -moz-crisp-edges; }\n"
        + "tt { white-space: pre; }\n"
        + "td { vertical-align: top }\n"
        + "@media print { .page-break { break-before: page; } }\n"
        + "</style><body><table>"
    )

    nmats = len(matrices)
    for i in range(nmats):
        params = matrices[i]
        matrixData = LoadMatrix(params.matrix)
        if matrixData is None:
            print(f"[{i + 1}/{nmats}] Matrix {i} with params {params} not loaded")
            continue

        params = FillDefaultParams(
            params,
            matrixData.matrix,
            matrixData.name,
            i,
            defaultChoiceParams,
        )
        print(
            f"[{i + 1}/{nmats}] {params.name} "
            f"{matrixData.matrix.shape[0]} x {matrixData.matrix.shape[0]}"
        )

        lines = CreateLine(matrixData.matrix, params, outDir, verbose)

        output += lines

    output += "</table></body></html>"

    with open(os.path.join(outDir, REPORT_NAME), "w") as f:
        f.write(output)


if __name__ == "__main__":
    argvs = list(sys.argv)

    if len(argvs) < 3:
        print("USAGE: matrix_report [matrix directory] [out directory]")
        exit(-1)

    matDir = argvs[1]
    outDir = argvs[2]
    matFnames = [
        f
        for f in os.listdir(matDir)
        if os.path.isfile(os.path.join(matDir, f))
        and os.path.splitext(f)[1] in [".mtx", ".npz"]
    ]

    matFnames.sort()

    CreateReport(
        matrices=[
            ReportParams(matrix=os.path.join(matDir, f), name=f) for f in matFnames
        ],
        outDir=outDir,
        verbose=True,
    )
