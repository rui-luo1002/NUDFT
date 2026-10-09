import nudft
from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *
from skimage import data, transform

nPix = 256

img = transform.resize(data.shepp_logan_phantom(), (nPix,nPix)).astype(complex128)

# derive Cartesian coord, load Spiral coord, Aera array
tupK_Cart = meshgrid(
    linspace(-nPix//2, nPix//2, nPix, endpoint=False),
    linspace(-nPix//2, nPix//2, nPix, endpoint=False),
    indexing='ij')[::-1]
arrK_Cart = array(tupK_Cart).transpose(1,2,0)
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
    arrSig[iPE,:] = nudft.dft(arrM0_Offres.reshape(nRO,-1), arrK_Cart.reshape(-1,2), arrK[iPE,:,:])

# reconstruct image using idft()
imgRec = nudft.idft((arrSig*arrAera).reshape(-1), arrK.reshape(-1,2), arrK_Cart.reshape(-1,2)).reshape(nPix,nPix)

# plot
figure()
subplot(121)
imshow(abs(img), cmap='gray')
title("Original Image")
subplot(122)
imshow(abs(imgRec), cmap='gray')
title("NUDFT Reconstruction")

savefig(__file__.replace(".py", ".png"), dpi=300)
# show()
