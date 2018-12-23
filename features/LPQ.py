import numpy as np
from scipy.signal import convolve2d
from scipy.cluster.vq import whiten

def LPQ(img, winSize=3, mode='nh'):
    STFTalpha = 1 / winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)

    convmode = 'valid'  # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img = np.float64(img)  # Convert np.image to double
    r = (winSize - 1) / 2  # Get radius from window size
    x = np.arange(-r, r + 1)[np.newaxis]  # Form spatial coordinates in window
    #  STFT uniform window
    #  Basic STFT filters
    w0 = np.ones_like(x)
    w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
    w2 = np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                          filterResp2.real, filterResp2.imag,
                          filterResp3.real, filterResp3.imag,
                          filterResp4.real, filterResp4.imag])
    # Whiten the freqResponse in order to remove cooleration between responses
    freqResp = whiten(freqResp)

    ## Perform quantization and compute LPQ codewords, basically that's a binary to decimal conversion
    ## because python
    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0) * (2 ** inds)).sum(2)
    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode == 'im':
        LPQdesc = np.uint8(LPQdesc)

    ## Histogram if needed
    if mode == 'nh' or mode == 'h':
        # use 256 bins, as a uint8 can only represent 0 -> 255
        # range(257) will reutrn 256 whcih wil be considered the right most border, so i will have a bin for every value
        # in the raneg of 0--> 255, where the last bin will contain all values in the closed range [255,256], which
        # translates to all values = 255
        LPQdesc, bin_edges = np.histogram(LPQdesc.flatten(), bins=range(257))
    ## Normalize histogram if needed
    if mode == 'nh':
        LPQdesc = LPQdesc / LPQdesc.sum()
    return LPQdesc
