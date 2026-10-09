import nudft
from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *

nPix = 256

# -------------------------------------------------
# Phantom parameters you can tune
# -------------------------------------------------
# Image coordinates run from -1 to 1 in both axes
# Circle 1: water
wat_center = (-0.3, 0.00)   # (x, y)
wat_radius  = 0.2
wat_value   = 1.0

# Circle 2: fat
fat_center = (0.3, 0.00)      # (x, y)
fat_radius  = 0.2
fat_value   = 1.0

# Background
bg_value = 0.0

# -------------------------------------------------
# Build custom phantom with two circles
# -------------------------------------------------
x = linspace(-1, 1, nPix, endpoint=False)
y = linspace(-1, 1, nPix, endpoint=False)
X, Y = meshgrid(x, y, indexing='ij')

img = full((nPix, nPix), bg_value, dtype=complex128)

wat_mask = (X - wat_center[0])**2 + (Y - wat_center[1])**2 <= wat_radius**2
fat_mask   = (X - fat_center[0])**2 + (Y - fat_center[1])**2 <= fat_radius**2

img[wat_mask] = wat_value
img[fat_mask]   = fat_value

# -------------------------------------------------
# derive Cartesian coord, load Spiral coord, Aera array
# -------------------------------------------------
tupK_Cart = meshgrid(
    linspace(-nPix//2, nPix//2, nPix, endpoint=False),
    linspace(-nPix//2, nPix//2, nPix, endpoint=False),
    indexing='ij')[::-1]
arrK_Cart = array(tupK_Cart).transpose(1, 2, 0)

arrK = asarray(load("./Resource/K.npy"))
arrAera = asarray(load("./Resource/Aera.npy"))
nPE, nRO, _ = arrK.shape

# -------------------------------------------------
# simulate image with offres effect
# fat only gets off-resonance phase
# -------------------------------------------------
arrOm = zeros([nPix, nPix], dtype=complex128)

# Apply off-resonance only to the fat circle
# wat_offres = (0e-6) * (2 * pi) * (42.58e6) * (3)   # water @ 3T
# fat_offres = (3.5e-6) * (2 * pi) * (42.58e6) * (3)   # fat @ 3T
# wat_offres = (-3.5e-6) * (2 * pi) * (42.58e6) * (3)   # water @ 3T
# fat_offres = (0e-6) * (2 * pi) * (42.58e6) * (3)   # fat @ 3T
wat_offres = (0e-6) * (2 * pi) * (42.58e6) * (3)   # water @ 3T
fat_offres = (0e-6) * (2 * pi) * (42.58e6) * (3)   # fat @ 3T

arrOm[wat_mask] = wat_offres
arrOm[fat_mask] = fat_offres

arrM0_Offres = array([img * exp(1j * t * arrOm) for t in 2.5e-6 * arange(nRO)])

print(arrM0_Offres.shape)  # test

# -------------------------------------------------
# simulate rawdata S using dft()
# -------------------------------------------------
arrSig = zeros([nPE, nRO], dtype=complex128)
for iPE in range(nPE):
    arrSig[iPE, :] = nudft.dft(
        arrM0_Offres.reshape(nRO, -1),
        arrK_Cart.reshape(-1, 2),
        arrK[iPE, :, :]
    )

# -------------------------------------------------
# reconstruct image using idft()
# -------------------------------------------------
imgRec = nudft.idft(
    (arrSig * arrAera).reshape(-1),
    arrK.reshape(-1, 2),
    arrK_Cart.reshape(-1, 2)
).reshape(nPix, nPix)

# -------------------------------------------------
# plot
# -------------------------------------------------

figure(figsize=(4,4))
imshow(abs(imgRec), cmap='gray', origin='lower')
axis('off')

savefig("./test/temp.png", dpi=300)

# figure(figsize=(10, 4))
# subplot(121)
# imshow(abs(img), cmap='gray', origin='lower')
# axis('off')

# subplot(122)
# imshow(abs(imgRec), cmap='gray', origin='lower')
# axis('off')

# savefig("./test/temp.png", dpi=300)

# show()