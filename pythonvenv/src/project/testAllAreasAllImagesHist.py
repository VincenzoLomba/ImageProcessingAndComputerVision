
# This file contains the raw code for the project. Let's experiment!
# Starting with all used imports
from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Defining the image lists for the first and second tasks (NOTICE: images are expected in grayscale)
actualDirectory = Path(__file__).resolve().parent
imagesPath = actualDirectory / 'data/images'
outputsPath = actualDirectory / 'data/images/outputs'
outputsPath.mkdir(exist_ok=True)
firstTaskImagesNames = ['Tesi00.bmp', 'Tesi01.bmp', 'Tesi12.bmp', 'Tesi21.bmp', 'Tesi31.bmp', 'Tesi33.bmp']
secondTaskImagesNames = {
    1: ['Tesi44.bmp', 'Tesi47.bmp', 'Tesi48.bmp', 'Tesi49.bmp'],
    2: ['Tesi50.bmp', 'Tesi51.bmp'],
    3: ['Tesi90.bmp', 'Tesi92.bmp', 'Tesi98.bmp']
}
realImagesNamesMap = {file.name.lower(): file.name for file in imagesPath.iterdir() if file.is_file() and file.suffix.lower() == '.bmp'}
firstTaskImagesRealNames = [realImagesNamesMap.get(name.lower()) for name in firstTaskImagesNames if name.lower() in realImagesNamesMap]
secondTaskImagesRealNames = {
    key: [realImagesNamesMap.get(name.lower()) for name in names if name.lower() in realImagesNamesMap]
    for key, names in secondTaskImagesNames.items()
}
if len(firstTaskImagesRealNames) != len(firstTaskImagesNames):
    missing = set(name.lower() for name in firstTaskImagesNames) - set(realImagesNamesMap.keys())
    raise FileNotFoundError(f"Missing images for the first task ({', '.join(missing)})")
for key, names in secondTaskImagesNames.items():
    missing = set(name.lower() for name in names) - set(realImagesNamesMap.keys())
    if missing:
        raise FileNotFoundError(f"Missing images for the second task (key {key}) ({', '.join(missing)})")

# Starting with T1 (in this task, ASSOMPTIONS labelled as strong are ones that are gonna be addressed in T2)

generateFiguresT1 = False

imagesT1 = [cv2.imread(str(imagesPath / name), cv2.IMREAD_GRAYSCALE) for name in firstTaskImagesRealNames]
i = 0
for image in imagesT1:
    
    size = image.shape
    i += 1
    print(f"\nImage {i} (size {size})")

    # Be aware: if our objective is to get rid of spurious white pixels in the BLOBs (present due to immulination issues),
    # gaussian filtering of the original image may be helpful as well as dangerous (thinning & altering rings thickness,
    # potentially breacking them, altering the topology of the BLOBs in vary bad way in terms of retrieving contours in a latter step)
    # This problematics (due to the usage of gaussian filtering) may be addressed performing closing on the latter obtained BLOBs.
    # But, a closing all alone solves in the first place the problem of spurious white pixels in the BLOBs,
    # so the initial step of gaussian filtering is indeed NOT useful.
    # Be aware: in case of an additioal objective of cleaning the image and getting rid of spurious noise in the BKG which is NOT RELATED to BLOBs,
    # pre-appending a gaussian filtering step to the pipeline may instead be very helpful (this is gonna be invetigated in Task2),
    # of course still mantaining the per-BLOB closing step to counteract the problematics of gaussian filtering on the BLOBs topology.
    #
    # k = 3
    # imageBlur = cv2.GaussianBlur(image, (k, k), 0)

    hist, bins = np.histogram(image.flatten(), bins = 256, range = [0,256])
    otsuThreshold, binaryImage = cv2.threshold(
        image,               # Input image (grayscale)
        0,                   # Threshold value (not used when using Otsu's method, so set to 0) 
        255,                 # Value to assign to pixels above the threshold
        cv2.THRESH_BINARY +  # First flag, indicates binary thresholding (default Flag specified here only for clarity)
        cv2.THRESH_OTSU      # Second flag, indicates relying on Otsu's method (to determine the optimal threshold value)
    )                        # BKG will be associated to 255 (white), FRG/BLOLB will be associated to 0 (black)
                             # (beacuse we're supposing that in the original image BKG is white-like and GRG/BLOBs are black-like)

    path = outputsPath / 'T1'
    path.mkdir(exist_ok=True)
    cv2.imwrite(str(path / firstTaskImagesNames[i-1]), binaryImage)

    if generateFiguresT1:
        plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Image {firstTaskImagesNames[i-1]} {image.shape}")
        plt.subplot(2,2,2)
        plt.imshow(binaryImage, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Binary {firstTaskImagesNames[i-1]} (Otsu's Threshold: {otsuThreshold})")
        plt.subplot(2,1,2)
        plt.title(f"Histogram of {firstTaskImagesNames[i-1]} with Otsu's Threshold {otsuThreshold}")
        plt.stem(hist)
        plt.axvline(x=otsuThreshold, color='r', linestyle='--', linewidth=2)
        plt.show()

    # Be aware: strong ASSUMPTION here, considering rods as the only BLOBs (connected components) present in the binary image
    # Be aware: let's use 8-connectivity due to some rods rings beeing potentally thin and/or broken ALIAS with diagonal blocks
    # Be aware: string ASSUMPTION, different BLOBs/rods are considerably separeted from each other
    # How the method behaves: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
    # What are the STATS: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5
    # Pay attention: in morphological operations, connectedComponentsWithStats and findContours...
    # ...BKG is considered to be 0 (black) and the FRG is considered to be non-zero
    numLabels, labelsImage, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(binaryImage), connectivity = 8)

    closingKernelSize = 3

    for BLOB in range(1, numLabels):  # Pay attention: we are starting from 1 to skip the background label (alias 0)

        STAT_LEFT, STAT_TOP, STAT_WIDTH, STAT_HEIGHT, STAT_AREA = stats[BLOB]
        ROI = labelsImage[STAT_TOP:STAT_TOP+STAT_HEIGHT, STAT_LEFT:STAT_LEFT+STAT_WIDTH].copy() # Be aware: labelsImage contains int32 numbers
        ROI[ROI != BLOB] = 0 # Getting rid of other BLOBs
        ROIheight, ROIwidth = ROI.shape
        padding = closingKernelSize // 2 # (padding the ROI to avoid defects in the latter closing morphological operation)
        paddedROI = np.zeros((ROIheight + 2*padding, ROIwidth + 2*padding), dtype=ROI.dtype)
        paddedROI[padding:ROIheight + padding, padding:ROIwidth + padding] = ROI
        paddedROI = (paddedROI != 0).astype(np.uint8) # Back to using uint8, that "morphologyEx" prefers (we've set 0 as BKG, 1 as FRG)
        top = STAT_TOP - padding   # Pay attention: MAY be negative
        left = STAT_LEFT - padding # Pay attention: MAY be negative
        closedROI = cv2.morphologyEx(paddedROI, cv2.MORPH_CLOSE, np.ones((closingKernelSize,closingKernelSize), np.uint8))

        # Given the assumptions written above, "closedROI" il a BLOB which is SURELY a rod AND...
        # ...without holes except for the expected ones (rings, one or two)
        # That means, using "findContours" on "closedROI" we SURELY get:
        # - one contour for the external border of the rod (the one we are interested in) (external contour)
        # - one contour for each ring (one or two) (internal contours)
        # And NO MORE contours (no noise, no other BLOBs, no holes except for the expected ones)!
        # How the method behaves: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
        # Retrival Modes: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
        # Contour Approximation Modes: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
        # Relying on algorithm: https://docs.opencv.org/4.x/d0/de3/citelist.html#CITEREF_suzuki85
        # NOTICE: "findContours" is based on the notion of 8-connectivity!
        contours, hierarchy = cv2.findContours(
            closedROI.copy(),           # binary imahe from which extract contours (0 as BKG, non-zero as FRG)
            cv2.RETR_CCOMP,             # this retrival mode makes retrived contours organized in externals and internals
                                        # in that case, hierarchy[0][i][3] (that contains the index of the parent contour) will be -1...
                                        # ...IFF contour i is an external one (the 0 index is necessary for hyerarchy-python-structure reasons)
            cv2.CHAIN_APPROX_NONE       # this approximation mode makes sure to retrive ALL the points of the contours (no approximation, no compression)
        )
        externalContours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
        internalContours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] != -1]

        # Be aware: the single contour (from either of the two lists) has shape (N, 1, 2)
        # => dealing on broadcasting, it can be summed to an array of shape (1, 1, 2)
        externalContoursOriginalImage = [contour + np.array([[[left, top]]]) for contour in externalContours]
        internalContoursOriginalImage = [contour + np.array([[[left, top]]]) for contour in internalContours]
        allContoursOriginalImage = externalContoursOriginalImage + internalContoursOriginalImage
        imageRGB = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for contour in allContoursOriginalImage:
            contourPoints = contour.reshape(-1, 2) # we pass from shape (N, 1, 2) to shape (N, 2)
            Xs = contourPoints[:, 0]
            Ys = contourPoints[:, 1]
            imageRGB[Ys, Xs] = [255, 0, 0]
        
        if generateFiguresT1:
            plt.figure()
            plt.imshow(imageRGB)
            plt.show()

# Proceeding with T2

generateFiguresT2 = False

allH = []

secondTaskImagesRealNamesFlatten = [name for namesList in secondTaskImagesRealNames.values() for name in namesList]
imagesT2 = [cv2.imread(str(imagesPath / name), cv2.IMREAD_GRAYSCALE) for name in secondTaskImagesRealNamesFlatten]
i = 0
for image in imagesT2:

    # W.r.t Task1 pipeline, here we are pre-appending a minimal invasive gaussian filtering on the original grayscale image,
    # with the main objective of cleaning the image and getting rid of spurious noise in the BKG which is NOT RELATED to BLOBs
    # (we are indeed addressing the problem "dirty inspection area due to the presence of scattered iron powder")
    # Cool remark: this gaussing filter also domesticate well scattered borders of screws-distractors
    k = 5
    image = cv2.GaussianBlur(image, (k, k), 0)

    size = image.shape
    i += 1
    print(f"\nImage {i} (size {size})")
    hist, bins = np.histogram(image.flatten(), bins = 256, range = [0,256])
    otsuThreshold, binaryImage = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    path = outputsPath / 'T2'
    path.mkdir(exist_ok=True)
    cv2.imwrite(str(path / secondTaskImagesRealNamesFlatten[i-1]), binaryImage)

    if generateFiguresT2:
        plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Image {secondTaskImagesRealNamesFlatten[i-1]} {image.shape}")
        plt.subplot(2,2,2)
        plt.imshow(binaryImage, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Binary {secondTaskImagesRealNamesFlatten[i-1]} (Otsu's Threshold: {otsuThreshold})")
        plt.subplot(2,1,2)
        plt.title(f"Histogram of {secondTaskImagesRealNamesFlatten[i-1]} with Otsu's Threshold {otsuThreshold}")
        plt.stem(hist)
        plt.axvline(x=otsuThreshold, color='r', linestyle='--', linewidth=2)
        plt.show()

    numLabels, labelsImage, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(binaryImage), connectivity = 8)

    # We add here a step in which we filter-out BLOBs with very small area compared to the rest,
    # considered to be spurious noise in the BKG which is NOT RELATED to BLOBs which is survived to the initial gaussian filtering step.
    # Important: why not avoid the initial gaussian filtering and directly realy on this methodology?
    # The main reason is that the initial gaussian filtering step is also very helpful to get rid of the part of this spurios noise that,
    # after segmentation-thresholding, would be connected to the BLOBs (and thus cannot be filtered-out with this area-based filtering step).

    areas = stats[:, cv2.CC_STAT_AREA]
    areas = areas[1:] # Excluding the area of the background label ("connectedComponentsWithStats" guarantees its position is "stats" is at index 0)
    maximumBlobArea = binaryImage.shape[0] * binaryImage.shape[1] # areas.max()
    hist, bins = np.histogram(areas, bins = maximumBlobArea+1, range = [0, maximumBlobArea+1])

    allH.append(hist)

    if generateFiguresT2:
        plt.figure()
        plt.title(f"Histogram of BLOBs areas in {secondTaskImagesRealNamesFlatten[i-1]}")
        plt.stem(hist)
        plt.show()

    closingKernelSize = 3

    for BLOB in range(1, numLabels):  # Pay attention: we are starting from 1 to skip the background label (alias 0)

        STAT_LEFT, STAT_TOP, STAT_WIDTH, STAT_HEIGHT, STAT_AREA = stats[BLOB]
        ROI = labelsImage[STAT_TOP:STAT_TOP+STAT_HEIGHT, STAT_LEFT:STAT_LEFT+STAT_WIDTH].copy() # Be aware: labelsImage contains int32 numbers
        ROI[ROI != BLOB] = 0 # Getting rid of other BLOBs
        ROIheight, ROIwidth = ROI.shape
        padding = closingKernelSize // 2 # (padding the ROI to avoid defects in the latter closing morphological operation)
        paddedROI = np.zeros((ROIheight + 2*padding, ROIwidth + 2*padding), dtype=ROI.dtype)
        paddedROI[padding:ROIheight + padding, padding:ROIwidth + padding] = ROI
        paddedROI = (paddedROI != 0).astype(np.uint8) # Back to using uint8, that "morphologyEx" prefers (we've set 0 as BKG, 1 as FRG)
        top = STAT_TOP - padding   # Pay attention: MAY be negative
        left = STAT_LEFT - padding # Pay attention: MAY be negative
        closedROI = cv2.morphologyEx(paddedROI, cv2.MORPH_CLOSE, np.ones((closingKernelSize,closingKernelSize), np.uint8))

        # Given the assumptions written above, "closedROI" il a BLOB which is SURELY a rod AND...
        # ...without holes except for the expected ones (rings, one or two)
        # That means, using "findContours" on "closedROI" we SURELY get:
        # - one contour for the external border of the rod (the one we are interested in) (external contour)
        # - one contour for each ring (one or two) (internal contours)
        # And NO MORE contours (no noise, no other BLOBs, no holes except for the expected ones)!
        # How the method behaves: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
        # Retrival Modes: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
        # Contour Approximation Modes: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
        # Relying on algorithm: https://docs.opencv.org/4.x/d0/de3/citelist.html#CITEREF_suzuki85
        # NOTICE: "findContours" is based on the notion of 8-connectivity!
        contours, hierarchy = cv2.findContours(
            closedROI.copy(),           # binary imahe from which extract contours (0 as BKG, non-zero as FRG)
            cv2.RETR_CCOMP,             # this retrival mode makes retrived contours organized in externals and internals
                                        # in that case, hierarchy[0][i][3] (that contains the index of the parent contour) will be -1...
                                        # ...IFF contour i is an external one (the 0 index is necessary for hyerarchy-python-structure reasons)
            cv2.CHAIN_APPROX_NONE       # this approximation mode makes sure to retrive ALL the points of the contours (no approximation, no compression)
        )
        externalContours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
        internalContours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] != -1]

        # Be aware: the single contour (from either of the two lists) has shape (N, 1, 2)
        # => dealing on broadcasting, it can be summed to an array of shape (1, 1, 2)
        externalContoursOriginalImage = [contour + np.array([[[left, top]]]) for contour in externalContours]
        internalContoursOriginalImage = [contour + np.array([[[left, top]]]) for contour in internalContours]
        allContoursOriginalImage = externalContoursOriginalImage + internalContoursOriginalImage
        imageRGB = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for contour in allContoursOriginalImage:
            contourPoints = contour.reshape(-1, 2) # we pass from shape (N, 1, 2) to shape (N, 2)
            Xs = contourPoints[:, 0]
            Ys = contourPoints[:, 1]
            imageRGB[Ys, Xs] = [255, 0, 0]
        
        if generateFiguresT2:
            plt.figure()
            plt.imshow(imageRGB)
            plt.show()

h = np.sum(allH, axis=0)
nonZeeroIndexes = np.nonzero(h)[0]
h = h[:nonZeeroIndexes[-1]+1]

threshold = 200
nonZeeroIndexesUnderThreshold = np.nonzero(h[:threshold])[0]
index = nonZeeroIndexesUnderThreshold[-1]
print("Max area: ", index)

plt.figure()
plt.stem(h)
plt.axvline(x=threshold, color='r', linestyle='--', linewidth=2)
plt.show()

h = h[:threshold]
plt.figure()
plt.stem(h)
plt.show()