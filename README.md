# myotube-alignment
 Code to compute the alignment of myotubes and nuclei in immunostainings

alignment_scatter.py -> Myotubes are detected first by smoothing and erosion, adaptive thresholding and watershed segmentation, labeling them and finding the orientation. Then, the nuclei are detected by Otsu thresholding, Gaussian blurring, erosion and adaptive thresholding. Then, watershed segmentation is applied, countours are found and their centroids calculated, obtaining the angle or orientation. 