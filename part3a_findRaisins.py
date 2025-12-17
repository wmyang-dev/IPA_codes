from skimage.measure import label, regionprops
from skimage.transform import rotate
from skimage.io import imshow, imread
from skimage.color import rgb2gray, label2rgb
from skimage.util import random_noise
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, remove_small_objects
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12,8)) #for display

def processing(input_img_grey, input_degree=0):
  threshold = threshold_otsu(input_img_grey) #apply otsu
  img_bin = input_img_grey < threshold # convert to binary

  border_cleared = clear_border(img_bin) #clear boarder
  # apply_closing = binary_closing(img_bin) #apply closing
  # border_cleared = clear_border(apply_closing) #clear boarder

  small_obj_removed = remove_small_objects(border_cleared)
  img_labeled = label(small_obj_removed)
  img_label_overlay = label2rgb(img_labeled, image=small_obj_removed)

  regions = regionprops(img_labeled)

  if input_degree == 0: #only printout at 0 degree(wihout noise and rotation) for reference
    for region in regions:
      print(f'Region Label: {region.label}, Area = {region.area}, Centroid = {region.centroid}, BBox = {region.bbox}')
    #BBox: Bounding Box Coordinates

  obj_num = len(regions) #get the number of found object
  print(f'There are {obj_num} objects found at {input_degree} degree') # printout number of found obj

  return img_bin, border_cleared, small_obj_removed ,img_label_overlay, obj_num

def displayForDebugging(input_grey_img, input_arr ,resize_img,specified_degree=0):
  #display the first element of array
  arr1_noised_rotated = rotate(input_grey_img, input_arr[0], order=1,resize=resize_img)
  arr1 = processing(arr1_noised_rotated, input_arr[0])
  arr1_overlay = arr1[3]
  ax[1,1].imshow(arr1_overlay, cmap='gray')
  ax[1,1].set_title(f'first element at {input_arr[0]} degree')
  # display median element of array
  arr_median_noised_rotated = rotate(input_grey_img, input_arr[len(input_arr)//2], order=1,resize=resize_img)
  arr_median = processing(arr_median_noised_rotated, input_arr[len(input_arr)//2])
  arr_median_overlay = arr_median[3]
  ax[1,2].imshow(arr_median_overlay, cmap='gray')
  ax[1,2].set_title(f'median element at {input_arr[len(input_arr)//2]} degree')
  # display last element of array
  arr_last_noised_rotated = rotate(input_grey_img, input_arr[len(input_arr)-1], order=1,resize=resize_img)
  arr_last = processing(arr_last_noised_rotated, input_arr[len(input_arr)-1])
  arr_last_overlay = arr_last[3]
  ax[1,3].imshow(arr_last_overlay, cmap='gray')
  ax[1,3].set_title(f'last element at {input_arr[len(input_arr)-1]} degree')

  # this is for testing , accepts a specified argument, by default turn-off
  if specified_degree != 0:
    arr_specified_noised_rotated = rotate(input_grey_img, specified_degree, order=1,resize=resize_img)
    arr_specified = processing(arr_specified_noised_rotated, specified_degree)
    arr_specified_overlay = arr_specified[3]
    ax[1,4].imshow(arr_specified_overlay, cmap='gray')
    ax[1,4].set_title(f'Display specified degree: {specified_degree} degree')


input_img = imread('raisins.jpg') #load the image
img_grey = rgb2gray(input_img) #convert to grey scale image

#------------Standard: without rotation and noise adding-----------
std_out_0_degree = processing(img_grey)
std_bin = std_out_0_degree[0]
std_border_cleared = std_out_0_degree[1]
std_sObj_removed = std_out_0_degree[2]
std_overlay = std_out_0_degree[3]
std_numOfObj_found = std_out_0_degree[4] 
#-----------------------------------------------------------------
#--------------Adding noise and apply rotation---------------------------
img_grey_noise = random_noise(img_grey, mode='gaussian')
# img_grey_noise = random_noise(input_img, mode='gaussian', mean=0.0, var=0.001) #enhance noise is not working, very sensitive to noise
failed_degrees = [] #store those degree unable to find same amount objects as std(without rotaion and noise added)
worked_degrees = []
resize_imgs = True
for degree in range(360):
  # img_noised_rotated = rotate(img_grey_noise, degree, order=1) #without resize the image
  img_noised_rotated = rotate(img_grey_noise, degree, order=1, resize=resize_imgs) #with resize the image
  img_to_process = processing(img_noised_rotated, degree)
  img_numOfObj_found = img_to_process[4]
  if img_numOfObj_found != std_numOfObj_found:
    failed_degrees.append(degree)
  else:
    worked_degrees.append(degree)

print(f'Failed degrees are: ', failed_degrees)
print(f'Numbers of failed degree: ', len(failed_degrees))
print(f'Numbers of worked degree: ', len(worked_degrees))
print(f'Without noise added and rotation found objects: ', std_numOfObj_found)
#-----------------------------------------------------------------

if len(failed_degrees) != 0:
  resize_imgs = False
  displayForDebugging(img_grey_noise, failed_degrees, resize_imgs)
else:
  resize_imgs = True
  displayForDebugging(img_grey_noise, worked_degrees,resize_imgs, 27)


#display the result of standard processing, without rotaion and noise adding
ax[0,0].imshow(input_img)
ax[0,0].set_title('Original image')

ax[0,1].imshow(img_grey, cmap='gray')
ax[0,1].set_title('Std greyscale image')

ax[0,2].imshow(std_bin, cmap='gray')
ax[0,2].set_title('Std binary image')

ax[0,3].imshow(std_overlay, cmap='gray')
ax[0,3].set_title('Overlay found objects')

ax[1,0].imshow(img_grey_noise, cmap='gray')
ax[1,0].set_title('Noise added')

for ax in ax.flatten():
  ax.axis('off')
plt.tight_layout()
plt.show()