from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.io import imshow, imread
from skimage.color import rgb2gray, label2rgb
from skimage.util import random_noise
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, remove_small_objects
from skimage.segmentation import clear_border
from skimage.transform import rotate
from skimage import filters, feature
import matplotlib.pyplot as plt
import numpy as np

# fig, ax = plt.subplots(nrows=4,ncols=4, figsize=(18,10)) 
#the first row images are input images, greyscale, ref_images...
#for half of display 0-180, every 15 degree = 12 images
#----------------------------------------------------------------
fig, ax = plt.subplots(nrows=6,ncols=6, figsize=(18,10)) # 360/36=10, display 36 processed image, run 10 times


def processing(input_img_grey, input_degree=0):
  threshold = threshold_otsu(input_img_grey) #apply otsu
  img_bin = input_img_grey < threshold # convert to binary

  # border_cleared = clear_border(img_bin) #clear boarder
  apply_closing = binary_closing(img_bin) #apply closing
  border_cleared = clear_border(apply_closing) #clear boarder

  small_obj_removed = remove_small_objects(border_cleared)
  img_labeled = label(small_obj_removed)
  img_label_overlay = label2rgb(img_labeled, image=small_obj_removed)

  regions = regionprops(img_labeled)

  # if input_degree == 0: #only printout at 0 degree(wihout noise and rotation) for reference
  print('Current degree: ', input_degree)
  for region in regions:
    print(f'Region Label: {region.label}, Area = {region.area}, Centroid = {region.centroid}, BBox = {region.bbox}')
    #BBox: Bounding Box Coordinates

  obj_num = len(regions) #get the number of found object
  print(f'There are {obj_num} objects found at {input_degree} degree') # printout number of found obj
  
  return img_bin, border_cleared, small_obj_removed ,img_label_overlay, obj_num , img_labeled

def areaCentroidProp(labelled_img):
  props_table = regionprops_table(
  labelled_img,
  properties=('area','centroid','orientation','bbox')
  )
  props_df = pd.DataFrame(props_table)

  min_area_index = props_df['area'].idxmin() #retrieve the index of smallest area
  min_area = props_df.loc[min_area_index, 'area'] #get the min area value

  centroid_y = round(props_df.loc[min_area_index, 'centroid-0'])
  centroid_x = round(props_df.loc[min_area_index, 'centroid-1'])

  minr = props_df.loc[min_area_index, 'bbox-0']
  minc = props_df.loc[min_area_index, 'bbox-1']
  maxr = props_df.loc[min_area_index, 'bbox-2']
  maxc = props_df.loc[min_area_index, 'bbox-3']

  #draw a rectangle
  bx = (minc, maxc, maxc, minc, minc)
  by = (minr, minr, maxr, maxr, minr)

  return centroid_y, centroid_x, bx, by, minr,minc, maxr, maxc, min_area

def displayAreaCentroid(input_prop, ref_img, degree,row, col):
  test_centroid_y = input_prop[0]
  test_centroid_x = input_prop[1]
  test_bx = input_prop[2]
  test_by = input_prop[3]
  test_minr = input_prop[4]
  test_minc = input_prop[5]
  test_maxr = input_prop[6]
  test_maxc = input_prop[7]
  test_min_area = input_prop[8]
  # display image
  ax[row, col].imshow(ref_img, cmap='gray')
  ax[row, col].set_title(f'{degree}: Centroid & Area')
  # draw a rectangle
  ax[row,col].plot(test_bx,test_by,'-r', linewidth=1.5)
  ax[row,col].plot(test_centroid_x, test_centroid_y, '.b', markersize=8)
  # highlight centroid, area
  # ax[row, col].text(test_maxr, test_maxc, f'Centoid:[{test_centroid_y}, {test_centroid_x}]', color='red', fontsize=10, ha='center', va='bottom')
  ax[row, col].text(30, 230, f'Centoid:[{test_centroid_y}, {test_centroid_x}]', color='red', fontsize=10, ha='center', va='bottom')
  # ax[row, col].text(test_maxr, test_minc, f'Area: {test_min_area}', color='red', fontsize=10, ha='center', va='bottom')
  ax[row, col].text(30, 100, f'Area: {test_min_area}', color='red', fontsize=10, ha='center', va='bottom')

input_img = imread('raisins.jpg')
# input_img_noise = random_noise(input_img) #add noise
# input_img_noise = random_noise(input_img, mode='gaussian', mean=0.1, var=0.06) #enhance noise
img_grey_0 = rgb2gray(input_img)
# img_grey_noise = random_noise(img_grey_0)
img_grey_noise = random_noise(img_grey_0, mode='gaussian', mean=0.0, var=0.02)
# img_grey_filtered = img_grey #apply filter lose details, closing is more robust here

#store elements every 15 degree to the array
# degrees = [degree for degree in range(15,195,15)] # the first half of 360 degree
# degrees = [degree for degree in range(195,360,15)] # the other half of 360 degree
degrees = [degree for degree in range(36)] # run the first 36 sets
# degrees = [degree for degree in range(36,72)] # run the second 36 sets
# degrees = [degree for degree in range(72,108)] # run the third 36 sets
# degrees = [degree for degree in range(108,144)] # run the fourth 36 sets
# degrees = [degree for degree in range(144,180)] # run the fifth 36 sets
# degrees = [degree for degree in range(180,216)] # run the sixth 36 sets
# degrees = [degree for degree in range(216,252)] # run the seventh 36 sets
# degrees = [degree for degree in range(252,288)] # run the eighth 36 sets
# degrees = [degree for degree in range(288,324)] # run the ninth 36 sets
# degrees = [degree for degree in range(324,360)] # run the last 36 sets

# for testing functionality only
# displayAreaCentroid(std_centroidArea,std_sObj_removed, 0, 0, 3) # 0 degree

degrees_index = 0 #pointer, will be moving while iterating through the array

for row in range(0,6):
  for col in range(0,6):
    # for degree in range(, 180, 15):
    degree = degrees[degrees_index]

    img_rotated = rotate(img_grey_noise, degree, order=1, resize=True)
    img_to_process = processing(img_rotated, degree)
    img_to_centroidArea = areaCentroidProp(img_to_process[5])
    displayAreaCentroid(img_to_centroidArea, img_to_process[2], degree, row, col)

    degrees_index +=1

for ax in ax.flatten():
  ax.axis('off')

plt.tight_layout()
plt.show()