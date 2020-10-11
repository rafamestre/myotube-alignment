# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:30:44 2017

@author: rmestre
"""


import skimage
import skimage.io as io
io.use_plugin('tifffile')
import cv2
from skimage import feature
import seaborn as sns
sns.set(style="ticks", context="talk", color_codes=True, palette="deep", 
        rc={"axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black",
            "text.color": "black","axes.labelcolor": "black"})
import scipy

from skimage.filters import threshold_otsu, threshold_adaptive, rank
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.morphology import disk, watershed
import pandas as pd
import matplotlib.patches as mpatches

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier
import os
import sys
from matplotlib import pyplot as plt
import numpy as np

from skimage.filters import sobel

import copy

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse, hough_circle, hough_circle_peaks
from skimage.draw import ellipse_perimeter, circle_perimeter
import math


####IMAGES ARE READ
currentDir = os.getcwd()

directoryImages = 'C:\\Users\\rmestre\\OneDrive - IBEC\\PhD\\Python\\Alignment\\'

folders = ['170518','170608','170704']

folderNb = 2
chosenFolder = folders[folderNb]

file = "Circle 25x 01 stack.tif"

file = "Square z stack 25 x4.tif"

file = "cut_image.tif"

file = 'Triangle 25x 01 same but in the middle stack.tif'


directorySave = directoryImages + chosenFolder + '\\'+file[:-4]+'\\'

if not os.path.exists(directorySave):
    os.makedirs(directorySave)


image_stack_complete = io.imread(directoryImages+chosenFolder+'\\'+file)


z_scale = 3.106 # µm per plane
xy_scale = 0.55 # µm per pixel

####WE TAKE ONLY THE MYOTUBES, WHICH ARE THE first STACK
if folderNb == 1:
    image_stack = image_stack_complete[:,:,:,0]
    z_size, y_size, x_size, channel_size = image_stack_complete.shape
elif folderNb == 0:
    image_stack = image_stack_complete[:,0]
    z_size, channel_size, x_size, y_size = image_stack_complete.shape
elif folderNb == 2:
    image_stack = image_stack_complete[:,:,:,0]
    z_size, y_size, x_size, channel_size = image_stack_complete.shape


boxes = list()

def draw_line(event, x, y, flags, params):
    global boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = list()
        print('Start Mouse Position: '+str(x)+', '+str(y))
        sbox = [x, y]
        boxes.append(sbox)
    elif event == cv2.EVENT_LBUTTONUP:
        print('End Mouse Position: '+str(x)+', '+str(y))
        ebox = [x, y]
        boxes.append(ebox)


img = image_stack[int(len(image_stack)/2)]
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 


cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_line)


while(1):
    cv2.imshow('image',img)
    if len(boxes) == 2:
        cv2.line(img,tuple(boxes[0]),tuple(boxes[1]),(255,0,0),5)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()

print(boxes)


x1 = boxes[0][0]
x2 = boxes[1][0]
y1 = boxes[0][1]
y2 = boxes[1][1]

referenceAngle = math.atan(np.abs(y2-y1)/np.abs(x2-x1))
referenceAngle = referenceAngle*180/np.pi

if x1 <= x2 and y1 <= y2:
    referenceAngle = 180 - referenceAngle
elif x2 <= x1 and y2 <= y1:
    referenceAngle = 180 - referenceAngle

print(referenceAngle)

def distanceToLine(P1,P2,P0):
    x1 = P1[0]
    y1 = P1[1]
    x2 = P2[0]
    y2 = P2[1]
    x0 = P0[1]
    y0 = P0[0]
    num = (y2-y1)*x0-(x2-x1)*y0+x2*y1 - y2*x1
    den = np.sqrt((y2-y1)**2+(x2-x1)**2)
    return np.abs(num/den)






"""To plot all the z-stack of images"""
nrows = np.int(np.ceil(np.sqrt(z_size)))
ncols = np.int(z_size // nrows + 1)


"""WE DETECT THE MYOTUBES

We apply smoothing and erosion to the image to remove noise and separate cells
from each other as best as possible"""

#Detection parameters
smooth_size = 5 # pixels
min_radius = 60
max_radius = 1000

"""We compute the threshold by calculating first the maximal integral projection
accross the whole z-stack. Then, we do an Otsu threshold according to all the
intensity values in the stack, and we call it global threshold"""
max_int_proj = image_stack.max(axis=0)
thresh_global = threshold_otsu(max_int_proj)

###Definitions of matrices
smoothed_stack = np.zeros_like(image_stack)
labeled_stack = smoothed_stack.copy()
distance = np.zeros_like(image_stack)
local_maxi = np.zeros_like(image_stack)
binary = np.zeros_like(image_stack)
markers =  np.zeros_like(image_stack)
th3 = np.zeros_like(image_stack)

kernel = np.ones((9,9),np.uint8)

## Labeling for each z plane:
for z, frame in enumerate(image_stack):
    
    smoothed = cv2.GaussianBlur(frame,(smooth_size,smooth_size),1)
    smoothed = cv2.erode(smoothed, kernel)

    smoothed_stack[z] = smoothed
    im_max = smoothed.max()
    thresh = thresh_global -20
    # thresh = threshold_otsu(smoothed)
    if im_max < thresh_global:
        labeled_stack[z] = np.zeros(smoothed.shape, dtype=np.int32)
        distance[z] = np.zeros_like(smoothed)
    else:
        binary[z] = smoothed > thresh
        
    th3[z] = cv2.adaptiveThreshold(smoothed,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,51,-15)
    
    distance[z] = ndimage.distance_transform_edt(th3[z])
    local_maxi[z] = peak_local_max(distance[z], min_distance=2*min_radius,exclude_border=False,
                                    indices=False, labels=smoothed)
    #markers[z] = ndimage.label(local_maxi[z])[0]
    ret,markers[z] = cv2.connectedComponents(th3[z])
    labeled_stack[z] = watershed(-distance[z], markers[z], mask=th3[z])
        





min_area = 1000
meanAngle = list()
stdAngle = list()
semAngle = list()
distanceList = list()
angles = list()

for z in range(z_size):
        
    new_labeled_stack = list()
    moments = list()
    angle = list()
    angle2 = list()
    aspect = list()
    centroid = list()
    dist = list()
    
    
    for region in regionprops(labeled_stack[z]):
        if region.area > min_area:
            moments.append(region.moments_central)
            
            aspect.append(region.major_axis_length/region.minor_axis_length)
            minr, minc, maxr, maxc = region.bbox
            if aspect[-1] > 2:
                new_labeled_stack.append(region)
                angle.append(region.orientation*360/(2*np.pi))
                centroid = region.centroid
                dist.append(distanceToLine((x1,y1),(x2,y2),centroid))
            
        
           
    for angNb in range(len(angle)):
        if angle[angNb] < 0:
            angle[angNb] += 180
        angle2.append(referenceAngle - angle[angNb])
        if angle2[-1] < - 90:
            angle2[-1] += 180
        elif angle2[-1] > 90:
            angle2[-1] -= 180
        
    meanAngle.append(np.mean(angle2))
    stdAngle.append(np.std(angle2))
    semAngle.append(np.std(angle2)/np.sqrt(len(angle2)))
    
    diff = 100

    angle3 = angle2[:]
    
    while diff >= 1:
        for ang in range(len(angle3)):
            if angle3[ang] > meanAngle[-1] + 90:
                angle3[ang] = angle3[ang] - 180
            elif angle3[ang] < meanAngle[-1] - 90:
                angle3[ang] = angle3[ang] + 180
                      
        meanBefore = meanAngle[-1]
        meanAngle[-1] = np.mean(angle3)
        diff = meanBefore - meanAngle[-1]
  
    
        
    stdAngle[-1] = np.std(angle3)
    semAngle[-1] = np.std(angle3)/np.sqrt(len(angle3))    
    distanceList.append(dist)
    angles.append(angle3)
    

zstack = [i*z_scale for i in range(z_size)]



####WE TAKE ONLY THE nuclei, WHICH ARE THE third STACK
if folderNb == 1:
    image_stack = image_stack_complete[:,:,:,2]
elif folderNb == 0:
    image_stack = image_stack_complete[:,1]
elif folderNb == 2:
    image_stack = image_stack_complete[:,:,:,2]



"""WE DETECT THE NUCLEI

We apply smoothing and erosion to the image to remove noise and separate cells
from each other as best as possible
We also define the minimum and maximum radius of the nuclei in pixels"""

#Detection parameters
smooth_size = 5 # pixels
min_radius = 60
max_radius = 150

"""We compute the threshold by calculating first the maximal integral projection
accross the whole z-stack. Then, we do an Otsu threshold according to all the
intensity values in the stack, and we call it global threshold"""
max_int_proj = image_stack.max(axis=0)
thresh_global = threshold_otsu(max_int_proj)

###Definitions of matrices
smoothed_stack = np.zeros_like(image_stack)
labeled_stack = smoothed_stack.copy()
distance = np.zeros_like(image_stack)
local_maxi = np.zeros_like(image_stack)
binary = np.zeros_like(image_stack)
markers =  np.zeros_like(image_stack)
th3 = np.zeros_like(image_stack)

kernel = np.ones((smooth_size,smooth_size),np.uint8)

## Labeling for each z plane:
for z, frame in enumerate(image_stack):
    
    smoothed = cv2.GaussianBlur(frame,(smooth_size,smooth_size),1)
    smoothed = cv2.erode(smoothed, kernel)
    #smoothed = rank.median(smoothed, disk(smooth_size))
    
    #smoothed = rank.enhance_contrast(smoothed, disk(smooth_size))
    smoothed_stack[z] = smoothed
    im_max = smoothed.max()
    thresh = thresh_global

        
    th3[z] = cv2.adaptiveThreshold(smoothed,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY,11,-10)
    
    #binary[z] = cv2.morphologyEx(binary[z], cv2.MORPH_CLOSE, kernel)
    
    distance[z] = ndimage.distance_transform_edt(th3[z])

    markers[z] = ndimage.label(local_maxi[z])[0]

    labeled_stack[z] = watershed(-distance[z], markers[z], mask=th3[z])
        
        


"""We find the contours of the images (edges that form a closed shape)"""
img_contour = np.zeros_like(image_stack)
contours = np.zeros_like(image_stack)


for z, frame in enumerate(smoothed_stack): 
    #binary[z] = cv2.morphologyEx(binary[z], cv2.MORPH_CLOSE, kernel)
    img_contour[z], contours, hierarchy = cv2.findContours(th3[z], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


"""We label the contours
Then, we check if the rectangles are too big (more than max_area)
If they are, we consider them separatedly and we apply further erosion to the
region, as they probably contain several cells. After that, we update the original
image with the patches of locally eroded areas"""
min_area = 100
max_area = 800

labeled_contours = [label(img_contour[z]) for z in range(z_size)]   
for z in range(z_size):
    bigBoxImage = list()
    bigBoxes = list()
    for region in regionprops(labeled_contours[z]):
        # Draw rectangle around segmented coins.
        if region.area  > max_area:
            bigBoxImage.append(region.filled_image )
            bigBoxes.append(region.bbox)


    kernel = np.ones((7,7), np.uint8)
    
    for rectangle in range(len(bigBoxImage)):
    
        im_erode = cv2.erode(bigBoxImage[rectangle].astype('uint8'), kernel,iterations=2)
        min_row, min_col, max_row, max_col = bigBoxes[rectangle]
        img_contour[z][min_row:max_row,min_col:max_col] = im_erode
               
               

"""We label again after eroding the big rectangles
We remove the rectangles with a small area"""
labeled_contours = [label(img_contour[z]) for z in range(z_size)]   
z = 1
new_labeled_stack = list()
meanAngle = list()
stdAngle = list()
meanTan = list()
stdTan = list()
semAngle = list()
distanceNuclei = list()
anglesNuclei = list()



for z in range(z_size):
    moments = list()
    angle = list()
    angle2 = list()
    angleTan = list()    
    centroid = list()
    dist = list()
    for region in regionprops(labeled_contours[z]):
        # Draw rectangle around segmented coins.
        if region.area  < min_area:
            continue
        minr, minc, maxr, maxc = region.bbox

        new_labeled_stack.append(region)
        moments.append(region.moments_central)
        angle.append(region.orientation*360/(2*np.pi))
        centroid = region.centroid
        dist.append(distanceToLine((x1,y1),(x2,y2),centroid))
        
    
    for angNb in range(len(angle)):
        if angle[angNb] < 0:
            angle[angNb] += 180
        angle2.append(referenceAngle - angle[angNb])
        if angle2[-1] < - 90:
            angle2[-1] += 180
        elif angle2[-1] > 90:
            angle2[-1] -= 180
        angleTan.append(math.tan(angle2[angNb]))
        
    meanTan.append(np.mean(angleTan))
    stdTan.append(np.mean(angleTan))
    
    meanAngle.append(np.mean(angle2))
    stdAngle.append(np.std(angle2))
    semAngle.append(np.std(angle2)/np.sqrt(len(angle2)))
    
    diff = 100
    angle3 = angle2[:]
    
    while diff >= 1:
        for ang in range(len(angle3)):
            if angle3[ang] > meanAngle[-1] + 90:
                angle3[ang] = angle3[ang] - 180
            elif angle3[ang] < meanAngle[-1] - 90:
                angle3[ang] = angle3[ang] + 180
                      
        meanBefore = meanAngle[-1]
        meanAngle[-1] = np.mean(angle3)
        diff = meanBefore - meanAngle[-1]
     
    
        
    stdAngle[-1] = np.std(angle3)
    semAngle[-1] = np.std(angle3)/np.sqrt(len(angle3))
    distanceNuclei.append(dist)
    anglesNuclei.append(angle3)
        
        
zstack = [i*z_scale for i in range(z_size)]


bin_length = 10
bin_max = np.int(np.max((x_size,y_size))*xy_scale)
bin_max = math.ceil(bin_max/bin_length)*bin_length
bins = np.linspace(0,np.int(bin_max),np.int(bin_max/bin_length),endpoint=False)
binsX = list()

binsMyotubes = list()
binsNuclei = list()

myotubesMean = list()
myotubesSEM = list()
myotubesstd = list()
nucleiMean = list()
nucleiSEM = list()
nucleistd = list()

z = 10

for i in range(len(bins)-1):
    binsX.append(bins[i] + bin_length/2)
    binsAuxMyotubes = list()
    binsAuxNuclei = list()
    for z in range(5,z_size):
        
        
        for j in range(len(distanceList[z])):
            if bins[i] <= distanceList[z][j]*xy_scale < bins[i+1]:
                binsAuxMyotubes.append(angles[z][j])
        
        for j in range(len(distanceNuclei[z])):
            if bins[i] <= distanceNuclei[z][j]*xy_scale < bins[i+1]:
                binsAuxNuclei.append(anglesNuclei[z][j])
    
    myotubesMean.append(np.mean(binsAuxMyotubes))
    if type(binsAuxMyotubes) is list:
        myotubesSEM.append(np.std(binsAuxMyotubes)/np.sqrt(len(binsAuxMyotubes)))
    else:
        myotubesSEM.append(0)
        
    nucleiMean.append(np.mean(binsAuxNuclei))
    if type(binsAuxNuclei) is list:
        nucleiSEM.append(np.std(binsAuxNuclei)/np.sqrt(len(binsAuxNuclei)))
    else:
        nucleiSEM.append(0)
        
    binsMyotubes.append(binsAuxMyotubes)
    binsNuclei.append(binsAuxNuclei)
        
    
    
        

maskMyotubes = np.isfinite(nucleiMean)

fig1 = plt.figure(figsize=(8,7))
xline = np.linspace(-100,np.max(binsX)+10)
plt.errorbar(binsX[:50],nucleiMean[:50],yerr=nucleiSEM[:50],capsize=5,markeredgewidth = 1, marker = 'o',
             color = "g", linestyle='dashed', markersize='5')
ax1 = fig1.axes[0]
ax1.set_xlabel("Distance from edge (µm)",fontsize=28)
ax1.set_title("Alignment with distance from edge",fontsize=28)
ax1.set_ylabel("Difference from reference angle (°)",fontsize=28)
ax1.set_ylim([-50,50])
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.tight_layout()
xmin, xmax = ax1.get_xlim()
plt.plot(xline,np.zeros_like(xline),linestyle='dashed',color='.10', linewidth = 1)
ax1.set_xlim([xmin,xmax])


fig2 = plt.figure(figsize=(8,7))
plt.errorbar(binsX,myotubesMean,yerr=myotubesSEM,capsize=5,markeredgewidth = 1, marker = 'o',
             color = "g", linestyle='dashed', markersize='5')
ax2 = fig2.axes[0]
ax2.set_xlabel("Distance from edge (µm)",fontsize=28)
ax2.set_title("Alignment in z stack",fontsize=28)
ax2.set_ylabel("Difference from reference angle (°)",fontsize=28)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
#plt.rcParams["figure.figsize"] = (8,8)
plt.tight_layout()


plt.figure()
plt.plot(binsX,[len(binsNuclei[y]) for y in range(0,len(binsX))])

plt.figure()
plt.plot(binsX,[len(binsMyotubes[y]) for y in range(0,len(binsX))])

f_val, p_val = scipy.stats.f_oneway(*binsNuclei[:50])

widths = [911,1240,1050,937,957,1100,605,466,1070,919,994,955,904,1010]

total = len(widths)

meanWidth = np.mean(widths)
stdWidth = np.std(widths)

sys.exit()







sys.exit()