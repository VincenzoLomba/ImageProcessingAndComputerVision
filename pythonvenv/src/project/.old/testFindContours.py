
# https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
# https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
# https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gae4156f04053c44f886e387cff0ef6e08

import numpy as np
import cv2
import matplotlib.pyplot as plt

image = "11111111," \
        "01110000," \
        "00001000," \
        "00010010," \
        "00010000," \
        "11111110," \
        "00000010," \
        "00000010"

mat = np.array([list(map(int, row)) for row in image.strip(',').split(',')])

contours, hierarchy = cv2.findContours(mat.astype(np.uint8),
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)

contour = contours[1]
z = np.zeros_like(mat, dtype=float)

plt.figure()

N = len(contour)

for i in range(N):
    x, y = contour[i][0]

    # Valore crescente tra 0 e 1
    z[y, x] = (i + 1) / N

    plt.clf()
    plt.subplot(1,2,1)
    plt.title("Originale")
    plt.imshow(mat, cmap='gray', vmin=0, vmax=1)

    plt.subplot(1,2,2)
    plt.title("Contorno progressivo")
    plt.imshow(z, cmap='gray', vmin=0, vmax=1)

    plt.pause(0.4)

plt.show()
