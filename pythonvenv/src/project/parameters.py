
# This Python file contains all the parameters used in the project.
# You can quickly change their values here, then observe the effects of these changes re-running the Jupyter Notebook file.
import numpy as np

# The expected common size (in pixels) of the working images
imagesExpectedShape = (255, 256)

# The path of this "parameters.py" file and the ABSOLUTE paths of the two Tasks images folders
from pathlib import Path
BASE_DIRECTORY = Path(__file__).resolve().parent
firstTaskImagesFolderPath = BASE_DIRECTORY / 'data/images/FirstTask'
secondTaskImagesFolderPath = BASE_DIRECTORY / 'data/images/SecondTask'

# The expected file extension of the working images (as a list of admissible extensions)
workingImagesExtension = ['bmp']

# Structuring element to be used for the erosion operation that is applyed to discern spurious internal holes from actual ones
structuringElement = np.ones((3, 3), np.uint8)

# Size S of the SxS Gaussian Filter kernel to be used for the smoothing operation applyed prior segmentation
filterKernelSize = 3

# The area threshold (in pixels) used to discern BLOBs not representative of actual objects (neither connectiong-rods nor distractors)
areaThreshold = 60

# The value (between 0 and 1/2) used to threshold the isotropicity measure of BLOBs C=λ1/(λ1+λ2)∈[0,1/2] (to get rid of distractor, alias BLOBs with a too high isotropic shape)
distractorsIsotropicityThreshold = 0.2

# This value specifies the size of the neighborhood (set to 2B+1) used to compute contours curvature (for each point)
B = 5

# The value (between 0 and 1/2) used to threshold the curvature measure C=λ1/(λ1+λ2)∈[0,1/2] of contours points (to individuate high-curvature points)
curvatureThreshold = 0.25