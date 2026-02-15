
# https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
# https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
# https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gae4156f04053c44f886e387cff0ef6e08
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = "11111111," \
        "01110000," \
        "00001000," \
        "00010010," \
        "00010000," \
        "11111110," \
        "10001010," \
        "11111010"

mat = np.array([list(map(int, row)) for row in image.strip(',').split(',')])

contours, hierarchy = cv2.findContours(mat.astype(np.uint8),
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)

contour = contours[0]
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
"""
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
        "10001010," \
        "11111010"

mat = np.array([list(map(int, row)) for row in image.strip(',').split(',')])

contours, hierarchy = cv2.findContours(mat.astype(np.uint8),
                                       cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_NONE)

if hierarchy is None or len(contours) == 0:
    raise RuntimeError("Nessun contorno trovato.")

hier = hierarchy[0]  # (n_contours, 4) -> [next, prev, child, parent]

# Esterni: parent == -1
external_idxs = [i for i in range(len(contours)) if hier[i][3] == -1]
# Interni: parent != -1
internal_idxs = [i for i in range(len(contours)) if hier[i][3] != -1]

plt.figure()

# Prima esterni, poi interni
order = [(idx, "esterno") for idx in external_idxs] + [(idx, "interno") for idx in internal_idxs]

min_val = 0.05  # <-- aggiunta: livello minimo, un filo meno nero

for idx, kind in order:
    contour = contours[idx]

    # Riparti da zero per ogni bordo
    z = np.zeros_like(mat, dtype=float)

    parent = hier[idx][3]
    if kind == "esterno":
        print(f"Inizio bordo ESTERNO (idx={idx})")
    else:
        print(f"Inizio bordo INTERNO (idx={idx}, parent={parent})")

    N = len(contour)

    for i in range(N):
        x, y = contour[i][0]

        # Valore crescente tra min_val e 1 (non parte da nero puro)
        z[y, x] = min_val + (1 - min_val) * ((i + 1) / N)

        plt.clf()
        plt.subplot(1,2,1)
        plt.title("Originale")
        plt.imshow(mat, cmap='gray', vmin=0, vmax=1)

        plt.subplot(1,2,2)
        plt.title("Contorno progressivo")
        plt.imshow(z, cmap='gray', vmin=0, vmax=1)

        plt.pause(0.4)

plt.show()
