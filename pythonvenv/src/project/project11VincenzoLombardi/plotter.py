
# This Python file encapsulates all the plotting functions used in the project (to keep the Jupyter Notebook file cleaner)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
from collections import defaultdict, Counter

def plotBinarizationResults(indices, imagesNames, images, binaryImages, histograms = None, otsuThresholds = None):
    """
    A simple method to plot the results of the binarization process.
    """
    withHistograms = (histograms is not None) and (otsuThresholds is not None)
    for index in indices:

        if withHistograms:
            plt.figure(figsize=(12, 3))
        else: 
            plt.figure(figsize=(10, 4))

        # Original image
        if withHistograms:
            plt.subplot(1, 4, 1)
        else: 
            plt.subplot(1, 2, 1)
        plt.imshow(images[index], cmap='gray', vmin=0, vmax=255)
        plt.title(f'Original image ({imagesNames[index]})')
        
        # Binary image
        if withHistograms:
            plt.subplot(1, 4, 2)
        else: 
            plt.subplot(1, 2, 2)
        plt.imshow(binaryImages[index], cmap='gray', vmin=0, vmax=255)
        plt.title(f'Binary image ({imagesNames[index]})')
        
        if withHistograms:
            # Histogram (bottom, full width)
            plt.subplot(1, 4, (3, 4))
            plt.bar(range(256), histograms[index], width=1)
            # _, _, baseLine = plt.stem(histograms[index])
            # plt.setp(baseLine, visible=False)
            plt.axvline(x=otsuThresholds[index], color='r', linestyle='--', label=f"Otsu's threshold: {otsuThresholds[index]}")
            plt.title(f'Gray-level Histogram ({imagesNames[index]})')
            plt.xlabel('Gray-level')
            plt.ylabel('Counts')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

def produceColorMap(N):
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html?utm_source=chatgpt.com
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
    colorsMap = plt.colormaps['Dark2'].resampled(N)
    return colorsMap, (np.array(
        [colorsMap(i)[:3] for i in range(N)]
    )*255).astype(np.uint8)


def plotImageConnectedComponents(
        image, binaryImage, BLOBs,
        values1 = None, threshold1 = None, histName1 = None, xLabel1 = None, yLabel1 = None,
        redBLOBs = None, orangeBLOBs = None,
        values2 = None, threshold2 = None, histName2 = None, xLabel2 = None, yLabel2 = None):
    """
    A simple method to plot the results of the connected components labeling process (for a SINGLE image, both for grayscale and binary versions).
    """
    withHistogram = (values1 is not None) and (threshold1 is not None) and (histName1 is not None) and (xLabel1 is not None) and (yLabel1 is not None)
    withDoubleHistogram = withHistogram and (values2 is not None) and (threshold2 is not None) and (histName2 is not None) and (xLabel2 is not None) and (yLabel2 is not None)
    
    if withHistogram:
        plt.figure(figsize=(12, 3))
    else: 
        plt.figure(figsize=(10, 4))

    colorsMap, colorsRGB = produceColorMap(len(BLOBs))
    imageRGB = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    binaryImageRGB = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2RGB)
    legendElements = []
    for index, BLOB in enumerate(BLOBs):
        ysROI, xsROI = np.where(BLOB.ROI != 0) # Retrieve row and column indexes of BLOB pixels within its ROI
        ys = ysROI + int(BLOB.STAT_TOP)
        xs = xsROI + int(BLOB.STAT_LEFT)
        # if not withHistogram:
            # imageRGB[ys, xs] = colorsRGB[index]
        binaryImageRGB[ys, xs] = colorsRGB[index]
        legendElements.append(
            mpatches.Patch(
                color=colorsMap(index),
                label=f"Label {BLOB.label}"
            )
        )
    if redBLOBs is not None:
        for BLOB in redBLOBs:
            ysROI, xsROI = np.where(BLOB.ROI != 0) # Retrieve row and column indexes of BLOB pixels within its ROI
            ys = ysROI + int(BLOB.STAT_TOP)
            xs = xsROI + int(BLOB.STAT_LEFT)
            binaryImageRGB[ys, xs] = (255, 0, 0)
    if orangeBLOBs is not None:
        for BLOB in orangeBLOBs:
            ysROI, xsROI = np.where(BLOB.ROI != 0) # Retrieve row and column indexes of BLOB pixels within its ROI
            ys = ysROI + int(BLOB.STAT_TOP)
            xs = xsROI + int(BLOB.STAT_LEFT)
            binaryImageRGB[ys, xs] = (255, 165, 0)

    if not withHistogram:
        plt.subplot(1, 2, 1)
        plt.title(f'Original image ({BLOBs[0].imageName})')
        # plt.title(f'Original image ({BLOBs[0].imageName}) with Rods BLOBs')
        # plt.legend(handles=legendElements)
    else: 
        plt.subplot(1, 4, 1)
        plt.title(f'Original image ({BLOBs[0].imageName})')
        # plt.legend(handles=legendElements, loc="upper left" if withDoubleHistogram else "lower left")
    plt.imshow(imageRGB)
    if not withHistogram:
        plt.subplot(1, 2, 2)
        plt.title(f'Binary image ({BLOBs[0].imageName}) with labeled BLOBs')
        plt.legend(handles=legendElements)
    else: 
        plt.subplot(1, 4, 2)
        plt.title(f'Binary image ({BLOBs[0].imageName})')
        plt.legend(handles=legendElements, loc="upper left" if withDoubleHistogram else "lower left")
    plt.imshow(binaryImageRGB)
    if withHistogram:
        if not withDoubleHistogram:
            plt.subplot(1, 4, (3,4))
        else:
            plt.subplot(1, 4, 3)
        # First histogram
        dictCounts = Counter(values1)
        sortedValues1, counts = zip(*sorted(dictCounts.items()))
        xAxisPositions = range(len(sortedValues1))
        plt.bar(xAxisPositions, counts)
        plt.xticks(xAxisPositions, sortedValues1) # Set explicit labels for x-axis values
        plt.title(f'{histName1}s Histogram ({BLOBs[0].imageName})')
        plt.xlabel(xLabel1)
        plt.ylabel(yLabel1)
        idx = sum(v < threshold1 for v in sortedValues1) # "v < threshold1" boolean interpreted as integer
        plt.axvline(x=idx-0.5, color='r', linestyle='--', label=f"{histName1} threshold: {threshold1}")
        if withDoubleHistogram: plt.ylim(0, max(dictCounts.values())+0.3)
        plt.legend()
        if withDoubleHistogram:
            plt.subplot(1, 4, 4)
            # Second histogram
            dictCounts = Counter(values2)
            sortedValues2, counts = zip(*sorted(dictCounts.items()))
            xAxisPositions = range(len(sortedValues2))
            plt.bar(xAxisPositions, counts)
            labels = [f"{x:.4f}" for x in sortedValues2]
            plt.xticks(xAxisPositions, labels) # Set explicit labels for x-axis values
            plt.title(f'{histName2}s Histogram ({BLOBs[0].imageName})')
            plt.xlabel(xLabel2)
            plt.ylabel(yLabel2)
            idx = sum(v < threshold2 for v in sortedValues2) # "v < threshold2" boolean interpreted as integer
            plt.axvline(x=idx-0.5, color='r', linestyle='--', label=f"{histName2} threshold: {threshold2}")
            plt.ylim(0, max(dictCounts.values())+0.3)
            plt.legend()
    plt.tight_layout()
    plt.show()

def plotSpuriousHolesFilled(originalROI, filledROI, invalidInternalContours):
    """
    A simple method to plot the results of the spurious holes filling process.
    """
    originalROIRGB = cv2.cvtColor(originalROI, cv2.COLOR_GRAY2RGB)
    filledROIRGB   = cv2.cvtColor(filledROI,   cv2.COLOR_GRAY2RGB)
    for invalidContour in invalidInternalContours:
        xs = invalidContour[:, 0, 0] # Expected shape of contour: (N, 1, 2)
        ys = invalidContour[:, 0, 1] # Expected shape of contour: (N, 1, 2)
        originalROIRGB[ys, xs] = (255, 0, 0)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original ROI (spurious internal contours in red)')
    plt.imshow(originalROIRGB)
    plt.subplot(1, 2, 2)
    plt.imshow(filledROIRGB)
    plt.title('ROI with filled spurious holes')
    plt.tight_layout()
    plt.show()

def plotBLOBAnalysis(imagesNames, images, BLOBs):
    """
    A simple method to plot the results of BLOB analysis and feartures extraction.
    """
    N = len(BLOBs)
    figureCols = 3
    figureRows = (N + figureCols - 1) // figureCols # Integer division WITH CEILING (-1 is indeed nedded for the said ceiling effect)
    figureH = 4*figureRows
    figureW = 12
    BLOBsDictionary = defaultdict(list) # Generate an empty dictionary that uses lists as values type
    for BLOB in BLOBs: BLOBsDictionary[BLOB.imageName].append(BLOB)
    maxBlobsPerImage = max((len(BLOBlist) for BLOBlist in BLOBsDictionary.values()), default = 0)
    colorsMap, colorsRGB = produceColorMap(maxBlobsPerImage)
    plt.figure(figsize=(figureW, figureH))
    for imageIndex, image in enumerate(images):
        name = imagesNames[imageIndex]
        imageRGB = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        legendElements = []
        for BLOBindex, BLOB in enumerate(BLOBsDictionary.get(name, [])):
            if BLOB.imageName != name: continue
            ysROI, xsROI = np.where(BLOB.ROI != 0) # Retrieve row and column indexes of BLOB pixels within its ROI
            ys = ysROI + int(BLOB.STAT_TOP)
            xs = xsROI + int(BLOB.STAT_LEFT)
            imageRGB[ys, xs] = colorsRGB[BLOBindex]
            # Connecting-rod type (A or B)
            legendElements.append(
                mpatches.Patch(
                    color=colorsMap(BLOBindex),
                    label=f"CRod {BLOB.label} (type {BLOB.type.name})"
                )
            )
            # Connecting-rod MER
            cx, cy = BLOB.centerBB
            theta = float(BLOB.orientationModuloPI)
            box = cv2.boxPoints(
                ((float(cx), float(cy)),
                (float(BLOB.length), float(BLOB.width)),
                float(np.rad2deg(theta)))
            )
            box = np.int32(np.round(box))
            cv2.polylines(imageRGB, [box], isClosed=True, color=tuple(int(v) for v in colorsRGB[BLOBindex]), thickness=1, lineType=cv2.LINE_8)
            # Connecting-rod position (alias barycenter/centroid)
            cx, cy = BLOB.centroid
            cv2.circle(imageRGB, (int(round(cx)), int(round(cy))), 3, (255, 0, 0), thickness=-1)
            # Connecting-rod orientation
            quarterL = 0.25 * float(BLOB.length)
            dx = quarterL * np.cos(theta)
            dy = quarterL * np.sin(theta)
            cv2.line(
                imageRGB,
                (int(round(cx-dx)), int(round(cy-dy))),
                (int(round(cx+dx)), int(round(cy+dy))),
                (255, 0, 0), thickness=1, lineType=cv2.LINE_8
            )
            # Connecting-rod width at the barycenter
            halfWb = 0.5 * float(BLOB.widthAtBarycenter)
            dx = halfWb * (-np.sin(theta))
            dy = halfWb * ( np.cos(theta))
            cv2.line(
                imageRGB,
                (int(round(cx-dx)), int(round(cy-dy))),
                (int(round(cx+dx)), int(round(cy+dy))),
                (255, 0, 0), thickness=1, lineType=cv2.LINE_4
            )
            # Connecting-rod contours
            imageRGB[BLOB.externalContour[:, 1], BLOB.externalContour[:, 0]] = (255, 0, 0)
            for internalContour in BLOB.internalContours:
                imageRGB[internalContour[:, 1], internalContour[:, 0]] = (255, 0, 0)
            # Holes centroids
            for (hx, hy) in BLOB.holesCenters:
                cv2.circle(imageRGB, (int(round(hx)), int(round(hy))), 3, (0, 0, 0), thickness=-1)
            # Holes diameters
            for (hx, hy), d in zip(BLOB.holesCenters, BLOB.holesDiameters):
                half = 0.5 * float(d)
                x1 = int(round(hx - half))
                x2 = int(round(hx + half))
                y  = int(round(hy))
                cv2.line(imageRGB, (x1, y), (x2, y), (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
        plt.subplot(figureRows, figureCols, imageIndex+1)
        plt.title(f'Image {name} BLOB analysis')
        plt.legend(handles=legendElements)
        plt.imshow(imageRGB)
    plt.tight_layout()
    plt.show()

def plotContoursEnhanced(imageName, image, blurredImage, binaryImageOld, binaryImageNew, BLOBs, holes, point, radius):
    """
    A simple method to plot an example (for a SINGLE image) of beneficial effects due to the usage of Gaussian filtering at the begin of the processing pipeline.
    """
    exampleBLOBs = ([b for b in BLOBs if b.imageName == imageName and len(b.internalContours) == holes])
    if len(exampleBLOBs) == 0: raise ValueError(f"No extracted BLOB with {holes} holes found for image '{imageName}'")
    xROI, yROI = int(exampleBLOBs[0].STAT_LEFT), int(exampleBLOBs[0].STAT_TOP)
    hROI, wROI = exampleBLOBs[0].ROI.shape
    ROIoriginal = image[yROI:yROI+hROI, xROI:xROI+wROI]
    ROIblurred = blurredImage[yROI:yROI+hROI, xROI:xROI+wROI]
    ROIbefore = binaryImageOld[yROI:yROI+hROI, xROI:xROI+wROI]
    ROIafter = binaryImageNew[yROI:yROI+hROI, xROI:xROI+wROI]

    originalRGB = cv2.cvtColor(ROIoriginal, cv2.COLOR_GRAY2RGB)
    blurredRGB = cv2.cvtColor(ROIblurred, cv2.COLOR_GRAY2RGB)
    beforeRGB = cv2.cvtColor(ROIbefore, cv2.COLOR_GRAY2RGB)
    afterRGB = cv2.cvtColor(ROIafter, cv2.COLOR_GRAY2RGB)

    cx, cy = int(point[0]), int(point[1])
    r = int(radius)
    cv2.circle(originalRGB, (cx, cy), r, (255, 0, 0), thickness=1, lineType=cv2.LINE_4)
    cv2.circle(blurredRGB, (cx, cy), r, (255, 0, 0), thickness=1, lineType=cv2.LINE_4)
    cv2.circle(beforeRGB, (cx, cy), r, (255, 0, 0), thickness=1, lineType=cv2.LINE_4)
    cv2.circle(afterRGB, (cx, cy), r, (255, 0, 0), thickness=1, lineType=cv2.LINE_4)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("Original ROI (unfiltered)")
    plt.imshow(originalRGB)
    plt.subplot(2, 2, 2)
    plt.title("Original ROI (filtered)")
    plt.imshow(blurredRGB)
    plt.subplot(2, 2, 3)
    plt.title("After segmentation (NO prior filtering)")
    plt.imshow(beforeRGB)
    plt.subplot(2, 2, 4)
    plt.title("After segmentation (with prior filtering)")
    plt.imshow(afterRGB)

    plt.tight_layout()
    plt.show()

def plotExternalContourCurvature(BLOB, externalContour, curvatureValues, curvatureThreshold, startingPoint):
    """
    A simple method to plot the curvature values computed along a contour.
    """
    contourToROI = externalContour - np.array([[BLOB.STAT_LEFT, BLOB.STAT_TOP]])
    RGBROI = cv2.cvtColor(BLOB.ROI, cv2.COLOR_GRAY2RGB)
    RGBROI[contourToROI[:, 1], contourToROI[:, 0]] = (0, 114, 189)
    cx, cy = int(startingPoint[0] - BLOB.STAT_LEFT), int(startingPoint[1] - BLOB.STAT_TOP)
    #r = int(1)
    #cv2.circle(RGBROI, (cx, cy), r, (255, 0, 0), thickness=-1, lineType=cv2.LINE_4)
    RGBROI[cy, cx] = (255, 0, 0)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax0.set_title(f"ROI (BLOB {BLOB.label} from image {BLOB.imageName})")
    ax0.imshow(RGBROI)
    ax1.set_title("Curvature values along the external contour")
    x = np.arange(len(curvatureValues))
    ax1.plot(x, curvatureValues)
    ax1.axhline(y=curvatureThreshold, color='r', linestyle='--', label=f"Curvature threshold: {curvatureThreshold}")
    ax1.set_xlabel("Contour point index")
    ax1.set_ylabel("Curvature value")
    ax1.grid(True, which='both', linewidth=0.5, alpha=0.3)
    ax1.set_ylim(0, 0.5)
    ax1.margins(x=0)
    ax1.legend()
    plt.tight_layout()
    plt.show()

def plotHighCurvatureCouples(imagesNames, images, highCurvatureCouples, BLOBs):
    indices = [i for i, name in enumerate(imagesNames) if name in highCurvatureCouples]
    N = len(indices)
    figureCols = 2
    figureRows = N
    figureH = 4*figureRows
    figureW = 10
    plt.figure(figsize=(figureW, figureH))
    subPlotIndex = 0
    for imageIndex in indices:
        iName = imagesNames[imageIndex]
        imageRGB = cv2.cvtColor(images[imageIndex], cv2.COLOR_GRAY2RGB)
        couplesReferences = [coupleReference for lst in highCurvatureCouples[iName].values() for coupleReference in lst]
        couples = [(contour1[idx1], contour2[idx2]) for (idx1, idx2, contour1, contour2) in couplesReferences]
        _, colorsRGB = produceColorMap(len(couples))
        for index, (point1, point2) in enumerate(couples):
            x1, y1 = int(point1[0]), int(point1[1])
            x2, y2 = int(point2[0]), int(point2[1])
            cv2.circle(imageRGB, (x1, y1), 4, color=tuple(int(v) for v in colorsRGB[index]), thickness=-1)
            cv2.circle(imageRGB, (x2, y2), 4, color=tuple(int(v) for v in colorsRGB[index]), thickness=-1)
            # cv2.line(imageRGB, (x1, y1), (x2, y2), (255, 0, 0), thickness=1, lineType=cv2.LINE_4)
        plt.subplot(figureRows, figureCols, subPlotIndex+1)
        plt.title(f"Original image ({iName})")
        plt.imshow(imageRGB)
        subPlotIndex += 1
        H, W = images[imageIndex].shape
        rgbImage = np.zeros((H, W, 3), dtype=np.uint8)
        counts = np.zeros((H, W), dtype=np.uint16)
        iBLOBs = [b for b in BLOBs if b.imageName == iName]
        _, colorsRGB = produceColorMap(len(iBLOBs))
        for BLOB in iBLOBs:
            ysROI, xsROI = np.where(BLOB.ROI != 0)
            ys = ysROI + int(BLOB.STAT_TOP)
            xs = xsROI + int(BLOB.STAT_LEFT)
            counts[ys, xs] += 1
        indexBLOB = 0
        overlappingBLOBs = counts > 1
        for BLOB in iBLOBs:
            ysROI, xsROI = np.where(BLOB.ROI != 0)
            ys = ysROI + int(BLOB.STAT_TOP)
            xs = xsROI + int(BLOB.STAT_LEFT)
            rgbImage[ys, xs] = colorsRGB[indexBLOB]
            indexBLOB += 1
        rgbImage[overlappingBLOBs] = (255, 0, 0) # red
        plt.subplot(figureRows, figureCols, subPlotIndex+1)
        plt.title(f"BLOB splitting results")
        plt.imshow(rgbImage)
        subPlotIndex += 1
    plt.tight_layout()
    plt.show()