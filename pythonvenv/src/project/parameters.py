
# This Python file contains all the parameters used in the project.
# You can quickly change their values here, then observe the effects of these changes re-running the Jupyter Notebook file.

# The expected common size (in pixels) of the working images
imagesExpectedShape = (255, 256)

# The path of this "parameters.py" file and the ABSOLUTE paths of the two tasks images folders
from pathlib import Path
BASE_DIRECTORY = Path(__file__).resolve().parent
firstTaskImagesFolderPath = BASE_DIRECTORY / 'data/images/FirstTask'
secondTaskImagesFolderPath = BASE_DIRECTORY / 'data/images/SecondTask'

# The expected file extension of the working images (as a list of admissible extensions), WITH the dot included
workingImagesExtension = ['bmp']

# Connectivity to be used for connected components labeling (8-way is suggested)
CONNECTIVITY = 8