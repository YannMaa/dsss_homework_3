import matplotlib.pyplot as plt
from PIL import Image
import json
import numpy as np


# Task 1

# load 4 pictures with their corresponding masks and metadata
image_0 = Image.open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\0.png")
image_0_seg = Image.open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\0_seg.png")
tmp_0 = open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\0.meta","r").read()
meta_0 = json.loads(tmp_0)

image_1 = Image.open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\1.png")
image_1_seg = Image.open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\1_seg.png")
tmp_1 = open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\1.meta","r").read()
meta_1 = json.loads(tmp_1)

image_2 = Image.open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\2.png")
image_2_seg = Image.open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\2_seg.png")
tmp_2 = open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\2.meta","r").read()
meta_2 = json.loads(tmp_2)

image_89 = Image.open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\89.png")
image_89_seg = Image.open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\89_seg.png")
tmp_89 = open(r"C:\Users\Yanni\dsss_homework_3\Mini_BAGLS_dataset\89.meta","r").read()
meta_89 = json.loads(tmp_89)


# Task 2

# overlay the images with the segmantation mask
image_0_overlay = image_0
image_0_overlay.paste(image_0_seg,(0,0),mask=image_0_seg)

image_1_overlay = image_1
image_1_overlay.paste(image_1_seg,(0,0),mask=image_1_seg)

image_2_overlay = image_2
image_2_overlay.paste(image_2_seg,(0,0),mask=image_2_seg)

image_89_overlay = image_89
image_89_overlay.paste(image_89_seg,(0,0),mask=image_89_seg)

# plotting
plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(image_0_overlay)
plt.title(meta_0.get("Subject disorder status"))
plt.subplot(2,2,2)
plt.imshow(image_1_overlay)
plt.title(meta_1.get("Subject disorder status"))
plt.subplot(2,2,3)
plt.imshow(image_2_overlay)
plt.title(meta_2.get("Subject disorder status"))
plt.subplot(2,2,4)
plt.imshow(image_89_overlay)
plt.title(meta_89.get("Subject disorder status"))

plt.show()


# Task 3

# load leaves 
leaves = Image.open(r"C:\Users\Yanni\dsss_homework_3\leaves.jpg")
leaves = np.asarray(leaves)

width, height,_= leaves.shape

# create empty arrays for storing the grayscale values
lightness_img = np.zeros(shape=(width,height))
luminosity_img = np.zeros(shape=(width,height))
average_img = np.zeros(shape=(width,height))

# grayscale methods
for x in range(width):
    for y in range(height):
        #get red, green and blue values from each pixel
        r,g,b = leaves[x,y,0],leaves[x,y,1],leaves[x,y,2]

        # lightness method
        min_rgb = np.min((r,g,b))/2.0
        max_rgb = np.max((r,g,b))/2.0
        lightness_img[x,y] = min_rgb+max_rgb

        # luminosity method
        luminosity_img[x,y] = 0.2989*r+0.5870*g+0.1140*b

        # average method
        average_img[x,y] = r/3.0+g/3.0+b/3.0


#plotting
plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(lightness_img, cmap='gray')
plt.title("Lightness Method")
plt.subplot(1,3,2)
plt.imshow(average_img, cmap='gray')
plt.title("Average Method")
plt.subplot(1,3,3)
plt.imshow(luminosity_img, cmap='gray')
plt.title("Luminosity Method")
plt.show()