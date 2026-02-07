import sys
import time
import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
import math
import dataclasses
from dataclasses import dataclass
from types import NoneType
from typing import Literal
from collections.abc import Callable
import importlib

JIT_ENABLED = False
GRAPH_ENABLED = False


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
        return numba.njit(f)


nx = None

if importlib.util.find_spec("networkx") is None:
    print("WARN: install networkx to use graph layouts")
else:
    import networkx as nx

    GRAPH_ENABLED = True


IMAGE_DIR = "matrix-images"
REPORT_NAME = "index.html"
DEFAULT_PX_COLS = 256
PLOT_MARGIN = 0.01
COL_PER_ROW_MAX = 2
AUTO_INCREASE_INT_SIZE = True
SEED = 42

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
    minColor: float = 1e-16
    maxColor: float = 1e5


def ScaleValue(colormap: ColormapParams, v: float):
    scale = colormap.scale
    minv = colormap.minColor
    maxv = colormap.maxColor

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

    pos = [v for v in values if v > colormap.zeroTol]
    neg = [v for v in values if v < -colormap.zeroTol]

    if len(values) == 0:
        return (1, 1, 1)

    if colormap.zeroTol == 0 and len(pos) + len(neg) == 0:
        return (1, 1, 0)  # yellow

    posAvg = sum(pos) / len(pos) if len(pos) > 0 else 0
    negAvg = sum(neg) / len(neg) if len(neg) > 0 else 0

    if posAvg < colormap.minColor and -negAvg < colormap.minColor:
        return (0, 1, 0)  # green

    red = ScaleValue(colormap, posAvg)
    blu = ScaleValue(colormap, -negAvg)

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


def SpringLayout(params, g):
    if GRAPH_ENABLED:
        return nx.spring_layout(g, seed=SEED)


@dataclass
class GraphParams:
    enabled: bool = True
    tol: float = -1
    layout = SpringLayout

    def Disabled():
        return SpectrumParams(enabled=False)


@dataclass
class ReportParams:
    matrix: MatrixDescType | MatrixLoader
    name: str | None = None

    colorParams: ColorParams | NoneType = None
    shadeParams: ShadeParams | NoneType = None
    scatterParams: ScatterParams | NoneType = None

    singularParams: SingularParams | NoneType = None
    condParams: CondParams | NoneType = None
    spectrumParams: SpectrumParams | NoneType = None

    blockParams: BlockParams | NoneType = None

    graphParams: GraphParams | NoneType = None


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

    defaultSingularParams: SingularParams = SingularParams()
    limitSingular: MatrixLimit = MatrixLimit.Square(n=10_000)

    defaultCondParams: CondParams = CondParams()
    limitCond: MatrixLimit = MatrixLimit(maxNNZ=100_000)

    defaultSpectrumParams: SpectrumParams = SpectrumParams()
    limitSpectrum: MatrixLimit = MatrixLimit.Square(n=5_000)

    defaultBlockParams: BlockParams = BlockParams()
    limitBlockParams: MatrixLimit = MatrixLimit()

    defaultGraphParams: GraphParams = GraphParams()
    limitGraphParams: MatrixLimit = MatrixLimit.Square(n=1_000, nnz=5_000)


BLOCKS_PRESET = DefaultChoiceParams(
    limitScatter=MatrixLimit.Disabled(),
    limitSingular=MatrixLimit.Disabled(),
    limitCond=MatrixLimit.Disabled(),
    limitSpectrum=MatrixLimit.Disabled(),
    limitGraphParams=MatrixLimit.Disabled(),
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

    if params.colorParams is None:
        params.colorParams = (
            defaultChoice.defaultColorParams
            if defaultChoice.limitColor.IsWithin(matrix)
            else ColorParams.Disabled()
        )

    if params.shadeParams is None:
        params.shadeParams = (
            defaultChoice.defaultShadeParams
            if defaultChoice.limitShade.IsWithin(matrix)
            else ShadeParams.Disabled()
        )

    if params.scatterParams is None:
        params.scatterParams = (
            defaultChoice.defaultScatterParams
            if defaultChoice.limitScatter.IsWithin(matrix)
            else ScatterParams.Disabled()
        )

    if params.singularParams is None:
        params.singularParams = (
            defaultChoice.defaultSingularParams
            if defaultChoice.limitSingular.IsWithin(matrix)
            else SingularParams.Disabled()
        )

    if params.condParams is None:
        params.condParams = (
            defaultChoice.defaultCondParams
            if defaultChoice.limitCond.IsWithin(matrix)
            else CondParams.Disabled()
        )

    if params.spectrumParams is None:
        if m == n:
            params.spectrumParams = (
                defaultChoice.defaultSpectrumParams
                if defaultChoice.limitSpectrum.IsWithin(matrix)
                else SpectrumParams.Disabled()
            )
        else:
            params.spectrumParams = SpectrumParams.Disabled()

    if params.spectrumParams.enabled and m != n:
        print(
            f"WARN: will not compute spectrum of {params.name}, "
            + "since it is not square"
        )
        params.spectrumParams = SpectrumParams.Disabled()

    if params.blockParams is None:
        params.blockParams = (
            defaultChoice.defaultBlockParams
            if defaultChoice.limitBlockParams.IsWithin(matrix)
            else BlockParams.Disabled()
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
            params.graphParams = (
                defaultChoice.defaultGraphParams
                if defaultChoice.limitGraphParams.IsWithin(matrix) and GRAPH_ENABLED
                else GraphParams.Disabled()
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


@jit
def CountKrc(
    m: int,
    n: int,
    rowPtr: np.ndarray,
    col: np.ndarray,
    values: np.ndarray,
    tol: float,
    maxR: int,
    maxC: int,
):
    krc = np.zeros((maxR, maxC))
    masks = np.zeros((maxR, maxC, n))

    lists = np.zeros((maxR, maxC, n))
    sizes = np.zeros((maxR, maxC))

    for i in range(m):
        start = rowPtr[i]
        end = rowPtr[i + 1]

        for rm in range(maxR):
            r = rm + 1
            if i % r == 0:
                for cm in range(maxC):
                    for k in range(int(sizes[rm, cm])):
                        masks[rm, cm, int(lists[rm, cm, k])] = 0
                    sizes[rm, cm] = 0

        for kk in range(end - start):
            k = start + kk
            j = col[k]
            v = values[k]

            if abs(v) <= tol:
                continue

            for rm in range(maxR):
                for cm in range(maxC):
                    jb = j // (cm + 1)

                    if masks[rm, cm, jb] == 0:
                        lists[rm, cm, int(sizes[rm, cm])] = jb
                        masks[rm, cm, jb] = 1
                        krc[rm, cm] += 1
                        sizes[rm, cm] += 1

    return krc


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
    verbose: bool

    def __init__(self):
        self.startTime = time.perf_counter()

    def StartSection(self, name):
        self.sectionTime = time.perf_counter()
        print(f"-> {name}")

    def FinishSection(self):
        now = time.perf_counter()
        ela = now - self.sectionTime
        print(f"   ({ela:6.2f} s)")

    def Finish(self):
        now = time.perf_counter()
        ela = now - self.startTime
        print(f"Finished all ({ela:6.2f} s)")


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

    out = HTMLOutput(params.name)

    out.AddData(
        "General data",
        f"{m} x {n}, nnz = {nnz} ({sparsity:8.4f}%)<br/>"
        + f"pos range = ({minPos:11.4e}, {maxPos:11.4e})<br/>"
        + f"neg range = ({minNeg:11.4e}, {maxNeg:11.4e})<br/>"
        + f"bandwidth = ({bandLower:11d}, {bandUpper:11d})<br/>",
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

        def Process(ib, jb, r, c, values):
            color = GetValuesColor(list(values), params.colorParams.colormap)
            imageData[ib, jb] = color

        ProcessPixelBlocks(pxRows, pxCols, matrix, Process)

        plt.imsave(os.path.join(outDir, filepath), imageData)

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
                    color = white + (red - white) * (z + 0.2)
                else:
                    color = white + (black - white) * z
                imageData[ib, jb] = color

        plt.imsave(os.path.join(outDir, filepath), imageData)

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
            xlabel="Columns",
            ylim=[-PLOT_MARGIN * (m - 1), (1 + PLOT_MARGIN) * (m - 1)],
            ylabel="Rows",
        )

        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(outDir, filepath), dpi=200, bbox_inches="tight")
        plt.close(fig)

        out.AddFigure("Scatter nonzeros", f'<img src="{filepath}" />')

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
        fig.savefig(os.path.join(outDir, filepath), dpi=200, bbox_inches="tight")
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

        out.AddData(
            "Approximate singular values",
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
        fig.savefig(os.path.join(outDir, filepath), dpi=200, bbox_inches="tight")
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

    # --- Block ---

    if params.blockParams.enabled:
        logger.StartSection("Block")

        filepath = imagePrefix + "-block.png"

        maxR = params.blockParams.maxR
        maxC = params.blockParams.maxC
        krc = CountKrc(
            m,
            n,
            matrix.indptr,
            matrix.indices,
            matrix.data,
            params.blockParams.tol,
            maxR,
            maxC,
        )

        lowerBounds = np.zeros((maxR, maxC))
        sc = params.blockParams.cacheSize
        sf = params.blockParams.floatSize
        si = params.blockParams.intSize

        for rm in range(maxR):
            for cm in range(maxC):
                r = rm + 1
                c = cm + 1
                bnnz = krc[rm, cm]

                lowerBounds[rm, cm] = (
                    bnnz * (r * c * sf + si)
                    + ((m + r - 1) // r + 1) * si
                    + m * sf
                    + n * sf
                ) / sc

        idx = np.argmin(lowerBounds)
        rm, cm = np.unravel_index(idx, lowerBounds.shape)
        minR = rm + 1
        minC = cm + 1
        minMiss = lowerBounds[rm, cm]

        lowerBounds = lowerBounds / minMiss - 1

        fig, ax = plt.subplots(figsize=(8, 8))

        ims = ax.imshow(
            lowerBounds,
            cmap="magma",
            extent=(0.5, maxR + 0.5, 0.5, maxC + 0.5),
            origin="lower",
        )

        ax.set(xlabel="Block cols", ylabel="Block rows")
        ax.invert_yaxis()

        fig.colorbar(ims, shrink=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(outDir, filepath), dpi=200, bbox_inches="tight")
        plt.close(fig)

        figN = out.AddFigure("Blocks", f'<img src="{filepath}" />')

        checkLines = ""

        for r, c in params.blockParams.checkRC:
            bnd = lowerBounds[r - 1, c - 1]
            checkLines += f"{r:2d} x {c:2d}: {bnd:11.4e} (+{bnd * 100:8.4f}%)<br/>"

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

        edgelist = np.zeros((nnz, 2))
        edgelistSize = 0
        for i in range(m):
            start = matrix.indptr[i]
            end = matrix.indptr[i + 1]
            for kk in range(end - start):
                k = start + kk
                j = matrix.indices[k]
                v = matrix.data[k]
                if abs(v) > params.graphParams.tol:
                    edgelist[edgelistSize] = (int(i), int(j))
                    edgelistSize += 1
        edgelist = edgelist[:edgelistSize]
        g = nx.from_edgelist(edgelist)
        layout = params.graphParams.layout
        pos = layout(g)

        posArray = np.zeros((2, n))

        for i in range(n):
            posArray[:, i] = pos[i]

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(posArray[0], posArray[1], c=np.arange(n), cmap="winter")

        for i1, i2, rest in nx.to_edgelist(g):
            i1 = int(i1)
            i2 = int(i2)
            if i1 != i2:
                ax.plot(
                    [posArray[0, i1], posArray[0, i2]],
                    [posArray[1, i1], posArray[1, i2]],
                    "k-",
                )

        ax.set(xticks=[], yticks=[])
        fig.tight_layout()
        fig.savefig(os.path.join(outDir, filepath), dpi=200, bbox_inches="tight")
        plt.close(fig)

        out.AddFigure("Graph", f'<img src="{filepath}" />')

        logger.FinishSection()

    # --- Tail ---

    line = out.FormLine()
    logger.Finish()

    return line


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
        #  defaultChoiceParams=BLOCKS_PRESET,
        outDir=outDir,
        verbose=True,
    )
