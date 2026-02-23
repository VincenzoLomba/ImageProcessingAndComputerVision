
# This Python file (imported by the Jupyter Notebook file) contains some miscellaneous functions used in the project.

import parameters

from enum import Enum
from dataclasses import dataclass
import numpy as np
import string
import cv2
import numpy as np
from collections import deque

class Connectivity(Enum): FOUR_CONNECTIVITY = 4; EIGHT_CONNECTIVITY = 8

class Task(Enum): FIRST_TASK = 1; SECOND_TASK = 2
def loadImages(task: Task):
    """
    A simple method to properly load working images (distingushing by task).
    Images are loaded in grayscale through OpenCV library usage.
    Two parallel lists are returned: a first one with images names (in lowercase) and a second one with the corresponding images as numpy arrays.
    """

    if task == Task.FIRST_TASK: imagesFolderPath = parameters.firstTaskImagesFolderPath
    elif task == Task.SECOND_TASK: imagesFolderPath = parameters.secondTaskImagesFolderPath
    else: raise ValueError(f"Unexpected Task value: '{task}'")
    if not imagesFolderPath.exists(): raise FileNotFoundError(f"The indicated images folder does not exist: '{imagesFolderPath}'")
    admittedExtensions = [extension.lower().strip(string.punctuation + string.whitespace) for extension in parameters.workingImagesExtension]
    # Notice: usage of strip (https://docs.python.org/3.6/library/stdtypes.html#str.strip) for better parsing safety
    imPths =  [
        filePath for filePath in imagesFolderPath.iterdir()
        if filePath.is_file() and filePath.suffix.lower().strip(string.punctuation + string.whitespace) in admittedExtensions
    ]
    if (len(imPths) == 0): raise FileNotFoundError(f"No images found in the indicated folder ('{imagesFolderPath}') with the indicated admissible extensions ({admittedExtensions})")
    imagesPaths = sorted(imPths, key=  lambda p: p.name.lower()) # Sorting only for visualization purposes within the Jupyter Notebook file, to later select images (to be shown to the reader) in a predictable way.
    names = []
    images = []
    for imagePath in imagesPaths:
        image = cv2.imread(str(imagePath), cv2.IMREAD_GRAYSCALE)
        # cv2.imread documentation: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gaffb68fce322c6e52841d7d9357b9ad2d
        if image is None or image.size == 0: raise OSError(f"Unexpected error while loading image: '{imagePath}'")
        if image.shape != parameters.imagesExpectedShape: raise ValueError(f"Unexpected image shape for image '{imagePath}'. Expected {parameters.imagesExpectedShape}, got {image.shape}")
        names.append(imagePath.name.lower())
        images.append(image)
    
    return names, images

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

def computeCovariance2D(points: np.ndarray):
    """
    A simple method that compute the covariance matrix for a set of 2D points,
    returning related eigenvalues/eigenvectors (in eigenvalues-ascending order, with normalized eigenvectors).
    """
    if points.ndim != 2 or points.shape[1] != 2: raise ValueError(f"Unexpected input shape (for the \"computeCovariance2D\" method). Expected (N, 2), got {points.shape}")
    N = points.shape[0]
    if N < 2: raise ValueError(f"At least 2 points required to compute a 2D covariance matrix (got {N}).")
    mu = points.mean(axis=0, keepdims=True) # keepdims=True used to maintain the shape (1, 2)
    pointsStar = points-mu
    covMatrix = pointsStar.T @ pointsStar / N
    eigenvalues, eigenvectors = np.linalg.eigh(covMatrix) # returns eigenvalues in ascending order and the corresponding normalized eigenvectors as columns (https://numpy.org/devdocs/reference/generated/numpy.linalg.eigh.html)
    return eigenvalues, eigenvectors

def breadthFirstSearchBFS(ROI, contours, startingPoint, endingPoint):
    """
    Implementation of a method to perform pathfinding (4-way connectivity) through a "search in amplitude" (indeed, breadth-first) strategy.
    Args:
    - ROI: region of interest in which perform the search (it should contain a single connected-component).
           This is a grid with 0-values for BKG and 255-values for FRG.
           The path will be searched as the minimum one (with breadth-first technique, alias "by levels") only living in FRG pixels.
    - contours: list of contours of the connected-component in the ROI (of course THEY MUST be referred to the ROI).
                All points in these contours will be considered as obstacles ones (as it is for all BKG points)
    - startingPoint: the point from which the path must start, expected to lie on a contour,
                     and also expected to be 4-way connected to at least one other FRG point which is NOT on any contour.
    - endingPoint: the point at which the path must end, expected to lie on a contour,
                   and also expected to be 4-way connected to at least one other FRG point which is NOT on any contour.
    """
    ROI = ROI.copy()
    h, w = ROI.shape
    # Generating obstacles map
    obstacles = np.zeros_like(ROI, dtype = np.uint8)
    for contour in contours:
        xs = contour[:,0]
        ys = contour[:,1]
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) # notice: & is element-wise between boolean arrays of same dimension
        if not np.all(valid): raise ValueError("Detected contours not referred to the ROI (some contour points are out of its bounds).")
        obstacles[ys, xs] = 255 # as usual, an image is indexed first with y and then with x
    # Defining (good) neighbors retriving method
    sx, sy = startingPoint
    ex, ey = endingPoint
    SEx = ex-sx; SEy = ey-sy # SE vector from starting point to ending point
    def retrieveNeighbors(point):
        x, y = point
        neighbors = []
        for nx, ny in ((x+1, y), (x, y-1), (x-1, y), (x, y+1)):
            if 0 <= nx < w and 0 <= ny < h and ROI[ny, nx] != 0 and obstacles[ny, nx] == 0:
                SNx = nx - startingPoint[0]; SNy = ny - startingPoint[1] # SN vector from starting point to the neighbor point
                distanceSigned = (SEx*SNy-SEy*SNx) # |SE⨯SN|=|SE|*|SN|*sinAngle(SE,SN)=|SEx*SNy-SEy*SNx|=|SE|*|SNdistanceFromSE| (|SE| is constant)
                distance2 = distanceSigned*distanceSigned
                neighbors.append(((nx, ny), distance2))
        # notice: the implemented BFS is slightly informed; in case of multiple pathes with the same lenght, we prefear the one that sticks more to the straight line between starting and ending points (alias sticks more to the SE vector)
        neighbors.sort(key=lambda el: el[1])
        return [n for n, _ in neighbors]
    # Initializing the endingList
    endingList = retrieveNeighbors(endingPoint)
    if len(endingList) == 0: raise ValueError("The ending point is NOT 4-way connected to at least one other FRG point which is NOT on any contour")
    # Initializing the alreadyVisitedList (as a matrix with the same shpae of ROI for better efficiency in accessing and updating it)
    alreadyVisitedListMatrix = np.zeros_like(ROI, dtype = np.uint8)
    # Initializing pointers dictionary
    pointers = {}
    # Initializing the openList (as a FIFO queue, accordingly to BFS logic)
    openLst = retrieveNeighbors(startingPoint)
    if len(openLst) == 0: raise ValueError("The starting point is NOT 4-way connected to at least one other FRG point which is NOT on any contour")
    openList = deque(openLst) # generates a FIFO
    for el in openList:
        alreadyVisitedListMatrix[el[1], el[0]] = 255 # adding the element to the alreadyVisitedList (notice: indexing with y and then with x)
        pointers[el] = startingPoint
    # Now simply running the BFS path search!
    success = False
    while len(openList) > 0:
        popped = openList.popleft() # retrieving the first element in the open list (FIFO) (to be expanded/closed)
        if popped in endingList:
            pointers[endingPoint] = popped
            success = True
            break
        for toBeVisited in retrieveNeighbors(popped):
            if alreadyVisitedListMatrix[toBeVisited[1], toBeVisited[0]] == 0: # not yet visited
                alreadyVisitedListMatrix[toBeVisited[1], toBeVisited[0]] = 255 # visiting it
                pointers[toBeVisited] = popped # pointing back to its visitor (the one just popped out and that is now visiting it)
                openList.append(toBeVisited) # adding the neighbor to the open list

    # Retrieving the path (if found)
    if not success: raise RuntimeError("No path found for the given input data")
    path = []
    traveller = endingPoint
    while traveller != startingPoint:
        path.append(traveller)
        traveller = pointers[traveller]
    path.append(startingPoint)
    return np.asarray(path, dtype=np.int32) # notice: the path is returned as reversed (from "endingPoint" to "startingPoint")

def search255OnContour(ROI, contour, startingIndex):
    """
    This method search for a 255-value point on a contour, starting from a given index and searching in both directions (forward and backward).
    The indices of the first two found points (both forward and backward) are returned.
    If no 255-value point is found, an exception is raised.
    Be aware that the passed contour must be referred in its points to the passed ROI.
    """
    N = len(contour)
    # Moving forward
    k = 0
    jf = 0
    while True:
        jf = (startingIndex + k) % N
        x = contour[jf, 0]
        y = contour[jf, 1]
        if ROI[y, x] == 255: break
        k += 1
        if k > N: raise RuntimeError("No 255-value point found in the whole contour")
    # Moving backward
    k = 0
    jb = 0
    while True:
        jb = (startingIndex - k) % N
        x = contour[jb, 0]
        y = contour[jb, 1]
        if ROI[y, x] == 255: break
        k += 1
        if k > N: raise RuntimeError("No 255-value point found in the whole contour")
    return jf, jb

def informedFloodFill(ROI, seed):
    """
    This method performs a flood fill starting from a seed point (that of course must be within the ROI), implementing the following policy:
    - 8-way connectivity is adopted
    - 0-valued points cannot be visited
    - 255-valued points can expand the flood-fill towards all neighbors that are NOT 0-valued
    - x-valued points (where x is neither 0 nor 255) can expand the flood-fill only towards neighbors that are valued in the same way (x-valued neighbors)
    """
    h, w = ROI.shape
    sx, sy = seed
    if not (0 <= sx < w and 0 <= sy < h): raise ValueError("The provided seed is out of ROI bounds")
    visited = np.zeros_like(ROI, dtype = np.uint8)
    if ROI[sy, sx] == 0: return visited
    def retrieveNeighbors(point):
        x, y = point
        neighbors = []
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1),(x+1,y+1),(x+1,y-1),(x-1,y+1),(x-1,y-1)):
            if 0 <= nx < w and 0 <= ny < h and ROI[ny, nx] != 0 and visited[ny, nx] == 0:
                if ROI[y, x] == 255 or ROI[ny, nx] == ROI[y, x]:
                    visited[ny, nx] = 255
                    neighbors.append((nx, ny))
        return neighbors
    openList = deque() # OpenList as a FIFO queue
    openList.append(seed)
    visited[sy, sx] = 255
    while len(openList) > 0:
        popped = openList.popleft()
        for toBeVisited in retrieveNeighbors(popped): openList.append(toBeVisited)
    return visited
