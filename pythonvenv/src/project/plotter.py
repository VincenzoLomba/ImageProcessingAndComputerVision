
# This Python file encapsulates all the plotting functions used in the project (to keep the Jupyter Notebook file cleaner)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
from collections import defaultdict

def plotBinarizationResults(indices, names, images, binaryImages, histograms = None, otsuThresholds = None):
    """
    A simple method to plot the results of the binarization process.
    For each image, a figure with three subplots is shown: the original image, its gray-level histogram and the corresponding binary image.
    Otsu's threshold is also shown in the histogram subplot as a vertical red line.
    """
    withHystograms = (histograms is not None) and (otsuThresholds is not None)
    for index in indices:

        if withHystograms:
            plt.figure(figsize=(12, 3))
        else: 
            plt.figure(figsize=(10, 4))

        # Original image
        if withHystograms:
            plt.subplot(1, 4, 1)
        else: 
            plt.subplot(1, 2, 1)
        plt.imshow(images[index], cmap='gray', vmin=0, vmax=255)
        plt.title(f'Original image ({names[index]})')
        
        # Binary image
        if withHystograms:
            plt.subplot(1, 4, 2)
        else: 
            plt.subplot(1, 2, 2)
        plt.imshow(binaryImages[index], cmap='gray', vmin=0, vmax=255)
        plt.title(f'Binary image ({names[index]})')
        
        if withHystograms:
            # Histogram (bottom, full width)
            plt.subplot(1, 4, (3, 4))
            plt.bar(range(256), histograms[index], width=1)
            #_, _, baseLine = plt.stem(histograms[index])
            #plt.setp(baseLine, visible=False)
            plt.axvline(x=otsuThresholds[index], color='r', linestyle='--', label=f"Otsu's threshold: {otsuThresholds[index]}")
            plt.title(f'Gray-level Histogram ({names[index]})')
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

def plotImageConnectedComponents(image, binaryImage, BLOBs):
    """
    A simple method to plot the results of the connected components labeling process (for a SINGLE image, both for grayscale and binary versions).
    """
    plt.figure(figsize=(10, 4))
    colorsMap, colorsRGB = produceColorMap(len(BLOBs))
    imageRGB = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    binaryImageRGB = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2RGB)
    legendElements = []
    for index, BLOB in enumerate(BLOBs):
        ysROI, xsROI = np.where(BLOB.ROI != 0) # Retrieve row and column indexes of BLOB pixels within its ROI
        ys = ysROI + int(BLOB.STAT_TOP)
        xs = xsROI + int(BLOB.STAT_LEFT)
        imageRGB[ys, xs] = colorsRGB[index]
        binaryImageRGB[ys, xs] = colorsRGB[index]
        legendElements.append(
            mpatches.Patch(
                color=colorsMap(index),
                label=f"Label {BLOB.label}"
            )
        )

    plt.subplot(1, 2, 1)
    plt.title(f'Original image ({BLOBs[0].imageName}) with Rods BLOBs')
    plt.legend(handles=legendElements)
    plt.imshow(imageRGB)
    plt.subplot(1, 2, 2)
    plt.title(f'Binary image ({BLOBs[0].imageName}) with Rods BLOBs')
    plt.legend(handles=legendElements)
    plt.imshow(binaryImageRGB)
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
            # Connecting-rod MER center
            cx, cy = BLOB.centerMER
            cv2.circle(imageRGB, (int(round(cx)), int(round(cy))), 3, (0, 0, 0), thickness=-1)
            # Connecting-rod MER
            box = cv2.boxPoints(((float(cx), float(cy)),
                     (float(BLOB.length), float(BLOB.width)),
                     float(np.rad2deg(BLOB.angleMER))))
            box = np.int32(np.round(box))
            cv2.polylines(imageRGB, [box], isClosed=True, color=tuple(int(v) for v in colorsRGB[BLOBindex]), thickness=1, lineType=cv2.LINE_8)
            # Connecting-rod position (alias barycenter/centroid)
            cx, cy = BLOB.centroid
            cv2.circle(imageRGB, (int(round(cx)), int(round(cy))), 3, (255, 0, 0), thickness=-1)
            # Connecting-rod orientation
            theta = float(BLOB.moduloPIorientation)
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
