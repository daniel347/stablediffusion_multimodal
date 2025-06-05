import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import metrics
from pathlib import Path

def mode_consistency_metric(colour, depth):
    grey = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    colour_edges = cv2.Canny(grey, 60, 120)
    depth_edges = cv2.Canny(depth, 7, 14)

    colour_edges_blur = cv2.GaussianBlur(colour_edges, (5,5), 1)
    depth_edges_blur = cv2.GaussianBlur(depth_edges, (5,5), 1)

    cosine_dist = np.sum(colour_edges_blur * depth_edges_blur)/np.sqrt(np.sum(colour_edges_blur**2) * np.sum(depth_edges_blur**2))

    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    ax[1][0].imshow(colour_edges)
    ax[1][0].set_title('Colour Edges')
    ax[1][1].imshow(depth_edges)
    ax[1][1].set_title('Depth Edges')
    ax[0][0].imshow(colour)
    ax[0][0].set_title('Colour')
    ax[0][1].imshow(depth)
    ax[0][1].set_title('Depth')
    fig.tight_layout()
    fig.savefig("canny_edges.png")
    plt.clf()

    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    ax[1][0].imshow(colour_edges_blur)
    ax[1][0].set_title('Colour Edges')
    ax[1][1].imshow(depth_edges_blur)
    ax[1][1].set_title('Depth Edges')
    ax[0][0].imshow(colour)
    ax[0][0].set_title('Colour')
    ax[0][1].imshow(depth)
    ax[0][1].set_title('Depth')
    fig.tight_layout()
    fig.savefig("canny_edges_blur.png")
    plt.clf()

    colour_edges = cv2.threshold(colour_edges, 100, 255, cv2.THRESH_BINARY)[1].astype(np.bool)
    depth_edges = cv2.threshold(depth_edges, 100, 255, cv2.THRESH_BINARY)[1].astype(np.bool)

    dist = metrics.hausdorff_distance(colour_edges, depth_edges, method="modified")
    return dist, cosine_dist


col_path = "../../datasets/lsun/colour/bedroom_254/778.jpg"
depth_path = "../../datasets/lsun/depth/bedroom_254/778.png"

col = cv2.imread(col_path, cv2.IMREAD_COLOR)
depth = (cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)//256).astype(np.uint8)

cols, rows = depth.shape[::-1]

dists = []
cos_dists = []
shifts = np.arange(20)

for i in shifts:
    M = np.float32([[1,0,i],[0,1,0]])
    depth_shift = cv2.warpAffine(depth,M,(cols,rows))
    d, cd = mode_consistency_metric(col, depth_shift)
    dists.append(d)
    cos_dists.append(cd)

plt.plot(shifts, dists)
plt.xlabel('Shift (pixels)')
plt.ylabel('Hausdorff Distance')
plt.title('Mode Consistency Metric')
plt.tight_layout()
plt.savefig("hausdorff_distance.png")
plt.clf()

plt.plot(shifts, cos_dists)
plt.xlabel('Shift (pixels)')
plt.ylabel('Cosine Distance')
plt.title('Mode Consistency Metric')
plt.tight_layout()
plt.savefig("cosine_distance.png")