
# This Python file encapsulates all the plotting functions used in the project (to keep the Jupyter Notebook file cleaner)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np

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

def plotImageConnectedComponents(image, binaryImage, BLOBs):
    """
    A simple method to plot the results of the connected components labeling process (for a SINGLE image, both for grayscale and binary versions).
    """
    plt.figure(figsize=(10, 4))
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html?utm_source=chatgpt.com
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
    colorsMap = plt.colormaps['Dark2'].resampled(len(BLOBs))
    colorsRGB = (np.array(
        [colorsMap(i)[:3] for i in range(len(BLOBs))]
    )*255).astype(np.uint8)
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
    plt.imshow(imageRGB)
    plt.legend(handles=legendElements)
    plt.subplot(1, 2, 2)
    plt.title(f'Binary image ({BLOBs[0].imageName}) with Rods BLOBs')
    plt.imshow(binaryImageRGB)
    plt.legend(handles=legendElements)
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