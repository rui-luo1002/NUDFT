import nudft
from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *
from skimage import data, transform
import finufft as fn

nPix, nAx = 256, 2

img = transform.resize(data.shepp_logan_phantom(), (nPix,nPix)).astype(complex128)

# derive Cartesian coord, load Spiral coord, Aera array
arrK = asarray(load("K.npy"))
arrAera = asarray(load("Aera.npy"))
nPE, nRO, _ = arrK.shape

# simulate image with offres effect
arrOm = zeros([nPix,nPix], dtype=complex128)
arrOm[:nPix//2,:] = (3.5e-6)*(2*pi)*(42.58e6)*(3) # fat@3T
arrM0_Offres = array([img*exp(1j*t*arrOm) for t in 2.5e-6*arange(nRO)])

print(arrM0_Offres.shape) # test

# simulate rawdata S using dft()
arrSig = zeros([nPE,nRO], dtype=complex128)
for iPE in range(nPE):
    arrSig[iPE] = nudft.nudft2d2(arrK[iPE], arrM0_Offres)

# reconstruct image using idft()
arrK = arrK[...,:nAx].reshape(-1,nAx)
arr2PiKTR = (2*pi*arrK.T)[::-1].copy(order="C")
imgRec = fn.nufft2d1(*arr2PiKTR, (arrSig*arrAera).flatten(), n_modes=(nPix,)*nAx)

# plot
figure(figsize=(10,5))
subplot(121)
imshow(abs(img), cmap='gray')
title("Original Image")
subplot(122)
imshow(abs(imgRec), cmap='gray')
title("NUDFT Reconstruction")
tight_layout()

savefig(__file__.replace(".py", ".png"), dpi=300)
# show()
