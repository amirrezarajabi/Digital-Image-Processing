# import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# read images as grayscale
img1 = cv.imread('Color_MRI.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('Color_MRI2.png', cv.IMREAD_GRAYSCALE)

"""
UI:
    choose 3 points on each image and save them in a list
"""
# create a copy of the images
img1_cpy = img1.copy()
img2_cpy = img2.copy()

# create a list to save the points
img1_points = []
img2_points = []

# create a function to save the points
def on_mouse(event, x, y, flags, params):
    # if left button is pressed
    if event == cv.EVENT_LBUTTONDOWN:
        # draw a circle on the image
        cv.circle(params[0], (x, y), 5, (255, 0, 0), -1)
        # show the image
        cv.imshow(params[1], params[0])
        # save the point in the list
        params[2].append([x, y])

# handle the first image
cv.namedWindow('Image 1')
cv.setMouseCallback('Image 1', on_mouse, (img1_cpy, 'Image 1', img1_points))
cv.imshow('Image 1', img1_cpy)

# handle the second image
cv.namedWindow('Image 2')
cv.setMouseCallback('Image 2', on_mouse, (img2_cpy, 'Image 2', img2_points))
cv.imshow('Image 2', img2_cpy)

# wait for the user to press esc
cv.waitKey(0)

# convert the lists to numpy arrays
img1_points = np.array(img1_points, dtype=np.float32)
img2_points = np.array(img2_points, dtype=np.float32)

# use opencv to find the affine transformation
affine_matrix = cv.getAffineTransform(img2_points, img1_points)

# apply the transformation to the second image
new_img = cv.warpAffine(img2, affine_matrix, (img2.shape[1], img2.shape[0]))


# get 2D histogram of img1 and img2 by binning them

def get2Dhistogram(img1, img2, bins=256):
    # create a 2D array to save the histogram
    hist = np.zeros((bins, bins), dtype=np.uint8)

    # loop over the image
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            # get the value of the pixel
            val1 = img1[i, j]
            val2 = img2[i, j]

            # add 1 to the corresponding bin
            hist[val1, val2] += 1

    return hist

# show the 2D histogram
def show2Dhistogram(img1, img2, title):
    # get the histogram
    hist = get2Dhistogram(img1, img2)

    # show the histogram
    plt.figure()
    plt.imshow(hist, cmap='gray')
    plt.title(title)
    plt.show()


# show the result of affine transformation
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img1, cmap='gray')
ax[0].set_title('Image 1')

ax[1].imshow(img2, cmap='gray')
ax[1].set_title('Image 2')

ax[2].imshow(new_img, cmap='gray')
ax[2].set_title('New Image')

plt.show()


# show the joint histogram

# joint histogram of the image 1 and the image 1
show2Dhistogram(img1, img1, 'Joint Histogram of Image 1 and Image 1')

# joint histogram of the image 1 and the image 2
show2Dhistogram(img1, img2, 'Joint Histogram of Image 1 and Image 2')

# joint histogram of the image 1 and the new image
show2Dhistogram(img1, new_img, 'Joint Histogram of Image 1 and New Image')

