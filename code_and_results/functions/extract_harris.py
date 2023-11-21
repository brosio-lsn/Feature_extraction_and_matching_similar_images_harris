import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter
import cv2
# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.04, thresh = 1e-6):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0
    
    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    #kernel for the x direction
    kernel_x = np.array([[0.5, 0, -0.5]])
    #compute gradients in x direction convolution with the kernel
    Ix = signal.convolve2d(img, kernel_x, mode='same', boundary='fill', fillvalue=0)
    #kernel for the y direction
    kernel_y = np.array([[0.5], [0], [-0.5]])
    #compute gradients in y direction convolution with the kernel
    Iy = signal.convolve2d(img, kernel_y, mode='same', boundary='fill', fillvalue=0)
    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    #compute the window gradients with a gaussian filter
    Ix2 = cv2.GaussianBlur(Ix**2, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    Iy2 = cv2.GaussianBlur(Iy**2, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    Ixy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    #compute the components of the matrix M (the fourth one M21 is equal to M12 as the matrix is symetric)
    M11 = Ix2
    M22 = Iy2
    M12 = Ixy
    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    #compute Harris response  =det(M) -k*trace(M)^2
    det_M = M11 * M22 - M12 * M12
    trace_M = M11 + M22
    C = det_M - k * (trace_M ** 2)
    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    #apply the threshold to obtain a mask
    filter = np.where(C > thresh, True, False)
    #apply the non-maximum suppression 
    max_filtered = ndimage.maximum_filter(C, size=3, mode ='constant', cval=0 )
    #create a mask (True value if the value is the maximum in the 3x3 neighborhood)
    max_mask = C == max_filtered
    #combine the two masks to obtain the final mask
    maximum_and_filter = max_mask & filter
    #apply the final mask to C to obtain corner coordinates ; BTW, for the coordinates, we onsider the origin of the axis as top left corner ; x axis is horizontal and y axis is vertical
    y,x =  np.where(maximum_and_filter) 
    #stack the coordinates to the correct output format
    corners = np.stack((x,y), axis=1)
    return corners, C

