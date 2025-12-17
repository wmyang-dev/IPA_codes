import numpy as np
from skimage.io import imshow, imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.feature import match_template
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import median

sample = imread('tie.jpg')
sample_g = rgb2gray(sample)
sample_g_noise = random_noise(sample_g) #adding noise
template_g = rgb2gray(imread('template_tie.jpg')) 
h_template_g, w_template_g = template_g.shape #height and width for template_g(grey scale)

median_filter_applied = median(sample_g_noise) #apply median filter to smooth the image
scores_list = [] #store the maxium score of current angle

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8))

for degree in range(360):
  median_g_rotated = rotate(median_filter_applied, degree, order=1, resize=True) #set order to 1 for bilinear, set resize=1 in case get clipped
  rotated_result = match_template(median_g_rotated, template_g) #matching the source image with template

  ij = np.unravel_index(np.argmax(rotated_result), rotated_result.shape)
  x_hor , y_ver = ij[::-1]
  scores_list.append(rotated_result.max()) #store the results score the highest at each degree to a list

  if rotated_result.max() >= np.max(scores_list):
    best_score = rotated_result.max()
    best_degree = degree
    # after found the best degree with best correlation, re-assign to image with rotation and template matching for output
    # instead of using the existed parameter for rotated image and matched template
    best_rotated_degree = rotate(median_filter_applied, best_degree, order=1, resize=True) #put the best degree has found for output display
    best_rotated_match = match_template(best_rotated_degree, template_g)

    print(f'The rotated_result has best score: {best_score:.2f}, at {degree} degree, located at [{x_hor},{y_ver}]')
    ax[1,2].add_patch(Rectangle((x_hor,y_ver), w_template_g, h_template_g, edgecolor='r', facecolor='none'))
    ax[1,2].text(x_hor,y_ver,f'{best_score:.2f}', color='red', fontsize=10, ha='center', va='bottom')
  else:
    print(f'Not-matching with score: {rotated_result.max():.2f} at degree: {degree}', f'located at [{x_hor},{y_ver}]')

ax[0,0].imshow(sample)
ax[0,0].set_title('Original image')

ax[0,1].imshow(sample_g, cmap='gray')
ax[0,1].set_title('Grey scale')

ax[0,2].imshow(sample_g_noise, cmap='gray')
ax[0,2].set_title('Noise added')

ax[1,0].imshow(median_filter_applied, cmap='gray')
ax[1,0].set_title('Median filter applied')

ax[1,1].imshow(best_rotated_match, cmap='gray')
ax[1,1].set_title(f'Template Matching at {best_degree} degree')

ax[1,2].imshow(best_rotated_degree, cmap='gray')
ax[1,2].set_title(f'Best matched instance at {best_degree} degree')

for ax in ax.flatten():
  ax.axis('off')
plt.show()
