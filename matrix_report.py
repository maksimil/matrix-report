import sys
import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
import math
import random
from dataclasses import dataclass
from types import NoneType
from typing import Literal
from collections.abc import Callable

IMAGE_DIR = "matrix-images"
REPORT_NAME = "index.html"

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
    pxCols: int = 256
    scale: Literal["log", "linear"] = "linear"

    def Disabled():
        return ShadeParams(enabled=False)


@dataclass
class ColormapParams:
    enabled: bool = True
    scale: Literal["log", "linear"] = "log"
    zeroTol: float = 0
    minColor: float = 1e-16
    maxColor: float = 1e20

    def Disabled():
        return ColormapParams(enabled=False)


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


def GetValuesColor(values: list[float] | float, colormap: ColormapParams):
    if type(values) is float:
        values = [values]

    pos = [v for v in values if v > 0]
    neg = [v for v in values if v < 0]

    if colormap.zeroTol == 0 and len(pos) + len(neg) == 0:
        return (1, 1, 0)  # yellow

    posAvg = sum(pos) / len(pos)
    negAvg = sum(neg) / len(neg)

    if posAvg < colormap.minColor and -negAvg < colormap.minColor:
        return (0, 1, 0)  # green

    red = ScaleValue(colormap, posAvg)
    blu = ScaleValue(colormap, negAvg)

    return (red, 0, blu)


@dataclass
class ColorParams:
    enabled: bool = True
    pxRows: int = -1
    pxCols: int = 256
    colorMap: ColormapParams = ColormapParams()

    def Disabled():
        return ColorParams(enabled=False)


@dataclass
class ReportParams:
    matrix: MatrixDescType | MatrixLoader
    name: str | None = None

    colorParams: ColorParams | NoneType = None
    shadeParams: ShadeParams | NoneType = None
    scatterParams: ColormapParams | NoneType = None

    computeSingular: bool | NoneType = None
    computeSpectrum: bool | NoneType = None
    computeCond: float | NoneType = None
    luCond: bool | NoneType = None
    enableFFT: bool | NoneType = None


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

    defaultScatterParams: ColormapParams = ColormapParams()
    limitScatter: MatrixLimit = MatrixLimit(maxNNZ=100_000)

    limitSingular: MatrixLimit = MatrixLimit.Square(n=10_000)

    limitSpectrum: MatrixLimit = MatrixLimit.Square(n=5_000)

    defaultCondTol: float = 1e-4
    limitCond: MatrixLimit = MatrixLimit(maxNNZ=100_000)

    limitLuCond: MatrixLimit = MatrixLimit.Disabled()

    limitFFT: MatrixLimit = MatrixLimit.Square(100_000)


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
    if params.name is None:
        if retName == "":
            params.name = f"Matrix {i}"
        else:
            params.name = retName

    if params.colorParams is None:
        if defaultChoice.limitColor.IsWithin(matrix):
            params.colorParams = defaultChoice.defaultColorParams
        else:
            params.colorParams = ColorParams.Disabled()

    if params.shadeParams is None:
        if defaultChoice.limitShade.IsWithin(matrix):
            params.shadeParams = defaultChoice.defaultShadeParams
        else:
            params.shadeParams = ShadeParams.Disabled()

    if params.scatterParams is None:
        if defaultChoice.limitScatter.IsWithin(matrix):
            params.scatterParams = defaultChoice.defaultScatterParams
        else:
            params.scatterParams = ColormapParams.Disabled()

    if params.computeSingular is None:
        params.computeSingular = defaultChoice.limitSingular.IsWithin(matrix)

    if params.computeSpectrum is None:
        params.computeSpectrum = defaultChoice.limitSpectrum.IsWithin(matrix)

    if params.computeCond is None:
        if defaultChoice.limitCond.IsWithin(matrix):
            params.computeCond = defaultChoice.defaultCondTol
        else:
            params.computeCond = -1.0

    if params.luCond is None:
        params.luCond = defaultChoice.limitLuCond.IsWithin(matrix)

    if params.enableFFT is None:
        params.enableFFT = defaultChoice.limitFFT.IsWithin(matrix)

    return params


def CreateLine(matrix: CSRMatrix, params: list[ReportParams], outDir: str):
    # --- Basic info ---
    m, n = matrix.shape
    nnz = matrix.getnnz()
    sparsity = nnz / (n * m) * 100

    posArr = matrix.data[matrix.data > 0]
    negArr = matrix.data[matrix.data < 0]
    minPos, maxPos, minNeg, maxNeg = math.nan, math.nan, math.nan, math.nan
    if len(posArr) > 0:
        minPos = posArr.min()
        maxPos = posArr.max()
    if len(negArr) > 0:
        minNeg = negArr.min()
        maxNeg = negArr.max()

    dataCell = (
        f"<tt><b>{params.name}</b><br/>"
        + f"{m} x {n}, nnz = {nnz} ({sparsity:8.4f}%)<br/>"
        + f"pos range = ({minPos:11.4e}, {maxPos:11.4e})<br/>"
        + f"neg range = ({minNeg:11.4e}, {maxNeg:11.4e})<br/>"
    )
    figCells = ""

    dataCell += "</tt>"

    return f"<tr><td>{dataCell}</td>{figCells}</tr>"


def CreateReport(
    matrices: list[ReportParams],
    outDir: str,
    defaultChoiceParams: DefaultChoiceParams = DefaultChoiceParams(),
):
    os.makedirs(os.path.join(outDir, IMAGE_DIR), exist_ok=True)

    print(f"Output dir: {outDir}, index file: {os.path.join(outDir, REPORT_NAME)}")

    output = (
        "<html><style>"
        + "img {{ width: 500; border: 1px solid black; }}\n"
        + ".pixelated { image-rendering: pixelated; "
        + "image-rendering: -moz-crisp-edges; }\n"
        + "tt {white-space: pre;}\n"
        + "@media print { .page-break { break-after: page; } }\n"
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

        line = CreateLine(matrixData.matrix, params, outDir)

        output += line

    output += "</table></body></html>"

    with open(os.path.join(outDir, REPORT_NAME), "w") as f:
        f.write(output)


__all__ = ["CreateReport"]

if __name__ == "__main__":
    argvs = list(sys.argv)

    if len(argvs) < 3:
        print("USAGE: matrix-report [matrix directory] [out directory]")
        exit(-1)

    matDir = argvs[1]
    outDir = argvs[2]
    matFnames = [
        f
        for f in os.listdir(matDir)
        if os.path.isfile(os.path.join(matDir, f))
        and os.path.splitext(f)[1] in [".mtx", ".npz"]
    ]

    CreateReport(
        matrices=[
            ReportParams(matrix=os.path.join(matDir, f), name=f) for f in matFnames
        ],
        outDir=outDir,
    )
