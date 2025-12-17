import numpy as np
from skimage.io import imshow, imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.feature import match_template
from skimage.feature import peak_local_max
from skimage.util import random_noise
from skimage.filters import median

sample = imread('tie.jpg')
sample_g = rgb2gray(sample)
sample_g_noise = random_noise(sample_g) #add noise
template_g = rgb2gray(imread('template_tie.jpg')) 

median_filter_applied = median(sample_g_noise) #apply median filter

result = match_template(median_filter_applied, template_g) #normalized cross-correlation

ij = np.unravel_index(np.argmax(result), result.shape)
#np.argmax: returns the index of the maximum value in the array.
x_hor, y_ver = ij[::-1] # x_hor correspond to column, y_ver correspond to row

h_temp_g, w_temp_g = template_g.shape

#draw the diagrams to display
fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(12,8))

ax[0,0].imshow(sample)
ax[0,0].set_title('Original image')

ax[0,1].imshow(sample_g, cmap='gray')
ax[0,1].set_title('Greyscale')

ax[0,2].imshow(sample_g_noise, cmap='gray')
ax[0,2].set_title('Noise added')

ax[1,0].imshow(median_filter_applied, cmap='gray')
ax[1,0].set_title('Median filter applied')

ax[1,1].imshow(result, cmap='gray')
ax[1,1].set_title('template matching', fontsize=15)

ax[1,2].imshow(median_filter_applied, cmap='gray')
ax[1,2].set_title('all matched instances', fontsize=15)

# threshold = result.max() * 0.74 #based on the global maxima
threshold = 0.7 #able to find all the seeking elements with this threshold, higher/lower will break
#tolerance approximately around 041~0.73, due to the random noise
#apply peak_local_max to rectangle the matched object
plm = peak_local_max(result, threshold_abs=threshold)

for row, col in plm:
  # the way to draw a rectangle is starting from the Upper-left corner
  ax[1,2].add_patch(Rectangle((col,row), w_temp_g, h_temp_g, edgecolor='r', facecolor='none'))
  score = result[row, col]
  ax[1,2].text(col, row, f'{score:.2f}', color='red', fontsize=6, ha='center', va='bottom')
plt.show()
