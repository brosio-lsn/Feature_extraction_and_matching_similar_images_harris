import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please

    # Calculate the squared differences using broadcasting
    squared_differences = (desc1[:, :, np.newaxis] - desc2.T) ** 2

    # Sum the squared differences along axis 1 (columns) to get the result
    return np.sum(squared_differences, axis=1)

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None

    #code that is common to the 3 methods : 
    #find the index of the minimum distance for each row (it find the nearest neighbor in image 2 for each descriptor of image 1)
    index_m2 = np.argmin(distances, axis=1)
    #create indices for each descriptor of image 1  
    index_m1 = np.arange(q1)
    #concatenate the indices to obtain the matches (we match the descriptor i of image 1 with its nearest neighbor in image 2)
    matches = np.concatenate((index_m1[:,np.newaxis], index_m2[:,np.newaxis]), axis=1)
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        #nothing more to do as we already have the matches
        matches = matches
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        #min indices in colums : => find for each descriptor of image 2 its nearest neighbor in image 1
        min_indices_columns = np.argmin(distances, axis=0)
        #create a mask for matches where the nearest neighbor principle is invertible 
        mask = np.arange(q1) == min_indices_columns[index_m2]
        #return the matches with applied match
        matches = matches[mask]
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        #keep only the matches row where the ratio between smallest and second smallest is inferior to 0.5:
        #find the second smallest value over a row
        second_mins = np.partition(distances, 1, axis=1)[:, 1]
        #find the smallest value over a row
        mins= np.min(distances, axis=1)
        #compute the ratio between the smallest and second smallest value
        ratio = mins/second_mins
        #create mask to keep only the matches where the ratio is inferior to 0.5
        mask = ratio < ratio_thresh
        #apply the mask to the matches
        matches = matches[mask]
    else:
        raise NotImplementedError
    return matches

