from skimage.measure import label, regionprops, regionprops_table
from skimage.io import imshow, imread
from skimage.color import rgb2gray, label2rgb
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.morphology import closing, remove_small_objects
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import numpy as np

input_img = imread('raisins.jpg') #load image
img_grey = rgb2gray(input_img)  #convert to greyscale

threshold = threshold_otsu(img_grey) #apply otsu threshold
img_bin = img_grey < threshold  #convert to binary

border_cleared = clear_border(img_bin) #clear border
closing_applied = closing(border_cleared) #apply closing
#--------------------------------------------------------------------
small_obj_removed = remove_small_objects(closing_applied) #remove leftovers

img_labeled = label(small_obj_removed) #labeling

img_label_overlay = label2rgb(img_labeled, image=small_obj_removed) #overlay found object
#-----------------------------------------------------------------

regions = regionprops(img_labeled) #apply regionprops for object information
# regions = regionprops(label_img)

for region in regions: #print out found object information
  print(f'Region Label: {region.label}, Area = {region.area}, Centroid = {region.centroid}, BBox = {region.bbox}')
#BBox: Bounding Box Coordinates

props_table = regionprops_table(
  img_labeled,
  properties=('area','centroid','orientation','bbox')
)

props_df = pd.DataFrame(props_table) #apply DataFrame, only shows on Jupyter

min_area_index = props_df['area'].idxmin() #retrieve the index of smallest area
min_area = props_df.loc[min_area_index, 'area'] #get the area value

centroid_y = round(props_df.loc[min_area_index, 'centroid-0'])
centroid_x = round(props_df.loc[min_area_index, 'centroid-1'])

minr = props_df.loc[min_area_index, 'bbox-0']
minc = props_df.loc[min_area_index, 'bbox-1']
maxr = props_df.loc[min_area_index, 'bbox-2']
maxc = props_df.loc[min_area_index, 'bbox-3']

#display output
fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(9,6))

ax[0,0].imshow(input_img)
ax[0,0].set_title('original image')

ax[0,1].imshow(img_grey, cmap='gray')
ax[0,1].set_title('grey image')


ax[0,2].imshow(border_cleared, cmap='gray')
ax[0,2].set_title('border_cleared')


ax[1,0].imshow(closing_applied, cmap='gray')
ax[1,0].set_title('apply closing')

ax[1,1].imshow(img_label_overlay, cmap='gray')
ax[1,1].set_title('overlay found object')

ax[1,2].imshow(small_obj_removed, cmap='gray')
ax[1,2].set_title('smallest raisin')
#draw a rectangle
bx = (minc, maxc, maxc, minc, minc)
by = (minr, minr, maxr, maxr, minr)
ax[1,2].plot(bx,by,'-r', linewidth=1.5)
ax[1,2].plot(centroid_x, centroid_y, '.b', markersize=8)
#highlight the centroid and area
ax[1,2].text(maxr, maxc, f'Centoid:[{centroid_y}, {centroid_x}]', color='red', fontsize=10, ha='center', va='bottom')
ax[1,2].text(maxr, minc, f'Area: {min_area}', color='red', fontsize=10, ha='center', va='bottom')

for ax in ax.flatten():
  ax.axis('off')

# print('small_obj_removed type: ', small_obj_removed.dtype)

plt.show()