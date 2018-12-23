from skimage.feature import local_binary_pattern
import numpy as np


def extract_lbp_features(image, points_number=8, radius=2, method='uniform'):
    """
        Performs:
            * Extract the LBP features
            * Get the histogram of LBP
            * Normalize using Min-Max

        Args:
            image: Binary image where background is white
            points_number: Number of points used to extract the LBP, default 8
            radius: Radius of the circular kernel used by LBP, default 2
            method: The method used by LBP, default 'uniform'

        Returns:
            Feature vector of shape (59,)

        Raises:
            None
    """
    BINS = 59
    LBP = local_binary_pattern(image=image, P=points_number, R=radius, method=method)
    hist = np.histogram(LBP, bins=np.arange(BINS))
    hist = ((hist[0] - hist[0].min()) / (hist[0].max() - hist[0].min()), hist[1])
    return hist[0]
