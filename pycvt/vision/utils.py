import numpy as np


def iou(gtboxes, dtboxes):
    '''numpy version of calculating IoU between two set of 2D bboxes.

    Args:
        gtboxes (np.ndarray): Shape (B,4) of ..,  4 present [x1,y1,x2,y2]
        dtboxes,np.ndarray,shape:(N,4), 4 present [x1,y1,x2,y2].

    Returns:
        np.ndarray: Shape (B,N)  .
    '''


    gtboxes = gtboxes[:, np.
                      newaxis, :]  #converse gtboxes:(B,4) to gtboxes:(B,1,4)
    ixmin = np.maximum(gtboxes[:, :, 0], dtboxes[:, 0])
    iymin = np.maximum(gtboxes[:, :, 1], dtboxes[:, 1])
    ixmax = np.minimum(gtboxes[:, :, 2], dtboxes[:, 2])
    iymax = np.minimum(gtboxes[:, :, 3], dtboxes[:, 3])
    intersection = (ixmax - ixmin + 1) * (iymax - iymin + 1)
    union = (gtboxes[:,:,2]-gtboxes[:,:,0]+1)*(gtboxes[:,:,3]-gtboxes[:,:,1]+1)\
            +(dtboxes[:,2]-dtboxes[:,0]+1)*(dtboxes[:,3]-dtboxes[:,1]+1)-intersection
    return intersection / union