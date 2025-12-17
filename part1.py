from skimage import io, filters , feature
from skimage.color import rgb2gray
from skimage.util import random_noise
from skimage.transform import resize
import matplotlib.pyplot as plt

# image_file = io.imread('selfie.jpg')
image_file = io.imread('selfie_glasses.jpg')
# image_file = io.imread('Lenna_(test_image).png')
ref_grey_scale = rgb2gray(image_file)
greyscale = rgb2gray(image_file)

#resize the image to 512*512
img_resized = resize(image_file,(512,512),anti_aliasing=True)
img_resized_grey = rgb2gray(img_resized)

print("Resized_image Shape: ", img_resized.shape)
print("Resized_image_grey Shape: ", img_resized_grey.shape)
print("Resized_image_grey Min/Max/Mean values: ", img_resized_grey.min(), greyscale.max(), greyscale.mean())

#add noise
noisy = random_noise(img_resized_grey, mode='gaussian', mean=0.0, var=0.01)

#sobel detector
sobel_applied = filters.sobel(noisy)

#canny detector
image = noisy
#apply two canny edge detector with different level of sigma
edges1 = feature.canny(image, sigma=2,low_threshold=0.1, high_threshold=0.3) #sigma is for gaussian smoothing
edges2 = feature.canny(image, sigma=3.3,low_threshold=0.1, high_threshold=0.3)

fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(12,8))

ax[0,0].imshow(greyscale, cmap=plt.cm.gray)
ax[0,0].set_title('Original image')
ax[0,0].axis('off')

ax[0,1].imshow(img_resized_grey, cmap=plt.cm.gray)
ax[0,1].set_title('resized_image:512*512')
ax[0,1].axis('off')

ax[0,2].imshow(noisy, cmap=plt.cm.gray)
ax[0,2].set_title('Add Gaussian noise')
ax[0,2].axis('off')

ax[0,3].imshow(sobel_applied, cmap=plt.cm.gray)
ax[0,3].set_title('Sobel_detector')
ax[0,3].axis('off')

ax[1,0].imshow(noisy, cmap='gray')
ax[1,0].set_title('Add Gaussian noise')
ax[1,0].axis('off')

ax[1,1].imshow(edges1, cmap='gray')
ax[1,1].set_title('canny edge1')
ax[1,1].axis('off')

ax[1,2].imshow(edges2, cmap='gray')
ax[1,2].set_title('canny edge2')
ax[1,2].axis('off')

# turn off axis
for a in ax.flatten():
  a.axis('off')

fig.tight_layout()

plt.savefig('output_comparison.jpg')

plt.show()

