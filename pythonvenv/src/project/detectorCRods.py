from enum import Enum
from dataclasses import dataclass
import numpy as np

class RodType(Enum): A = 1; B = 2
@dataclass
class CRodBLOB:
    imageName: str
    label: int
    STAT_LEFT: np.int32
    STAT_TOP: np.int32
    STAT_AREA: np.int32
    ROI: np.typing.NDArray[np.uint8]
    centroid: tuple[np.float64, np.float64]
    externalContour: np.typing.NDArray[np.int32] = None # nparray of shape (N, 2)
    internalContours: list[np.typing.NDArray[np.int32]] = None # list of nparrays of shape (N, 2)
    type: RodType = None
    orientationModuloPI: np.float64 = None
    length: np.float64 = None
    width: np.float64 = None
    centerBB: tuple[float, float] = None
    widthAtBarycenter: np.float64 = None
    holesCenters: list[tuple[np.float64, np.float64]] = None
    holesDiameters: list[np.float64] = None
    

