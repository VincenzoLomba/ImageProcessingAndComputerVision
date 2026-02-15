
# This Python file (imported by the Jupyter Notebook file) contains some miscellaneous functions used in the project.

from enum import Enum
import parameters
import string
import cv2
import numpy as np
from dataclasses import dataclass

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
        names.append(imagePath.name.lower())
        images.append(image)
    
    return names, images

class RodType(Enum): A = 1; B = 2
@dataclass
class RodBLOB:
    imageName: str
    label: int
    STAT_LEFT: np.int32
    STAT_TOP: np.int32
    STAT_AREA: np.int32
    ROI: np.typing.NDArray[np.uint8]
    centroid: tuple[np.float64, np.float64]
    type: RodType = None
    moduloPIorientation: np.float64 = None
    length: np.float64 = None
    width: np.float64 = None
    widthAtBarycenter: np.float64 = None
    holesCenters: list[tuple[np.float64, np.float64]] = None
    holesDiameters: list[np.float64] = None
    

