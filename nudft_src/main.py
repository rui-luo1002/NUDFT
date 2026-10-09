from numpy import *
from numpy.typing import *
from typing import *
from .ext import _dft

def _checkDim(dimDataSrc:NDArray, dimCoorSrc:NDArray, dimCoorDst:NDArray) -> bool:
    if \
    (
        dimDataSrc.size != 2 or
        dimCoorSrc.size != 2 or
        dimCoorDst.size != 2 or
        dimDataSrc[1] != dimCoorSrc[0] or # Npt consistency
        dimDataSrc[0] not in [1, dimCoorDst[0]] or # Nt consistency
        dimCoorSrc[1] != dimCoorDst[1] # Ndim consistency
    ):
        return False
    else:
        return True
    
def dft(arrSrc:NDArray, arrCoorSrc:NDArray, arrCoorDst:NDArray) -> NDArray:
    """
    Perform multi-frequency DFT.

    Args:
        arrSrc (NDArray): source data, shape: (lenDst, lenSrc) or (lenSrc,)
        arrCoorSrc (NDArray): source coordinates, shape: (lenSrc, nAx)
        arrCoorDst (NDArray): destination coordinates, shape: (lenDst, nAx)

    Returns:
        NDArray: DFT result, shape: (lenDst,)
    """
    if arrSrc.ndim == 1: arrSrc = arrSrc[newaxis,:]
    if not _checkDim(array(arrSrc.shape), array(arrCoorSrc.shape), array(arrCoorDst.shape)): raise RuntimeError("")
    return _dft(arrSrc, arrCoorSrc, arrCoorDst, 0)

def idft(arrSrc:NDArray, arrCoorSrc:NDArray, arrCoorDst:NDArray) -> NDArray:
    """
    Perform multi-frequency IDFT.

    Args:
        arrSrc (NDArray): source data, shape: (lenDst, lenSrc) or (lenSrc,)
        arrCoorSrc (NDArray): source coordinates, shape: (lenSrc, nAx)
        arrCoorDst (NDArray): destination coordinates, shape: (lenDst, nAx)

    Returns:
        NDArray: IDFT result, shape: (lenDst,)
    """
    if arrSrc.ndim == 1: arrSrc = arrSrc[newaxis,:]
    if not _checkDim(array(arrSrc.shape), array(arrCoorSrc.shape), array(arrCoorDst.shape)): raise RuntimeError("")
    return _dft(arrSrc, arrCoorSrc, arrCoorDst, 1)

def kCartesian(n_modes:Tuple[int,...]) -> NDArray:
    y = arange(-(n_modes[0]//2), (n_modes[0]+1)//2)
    x = arange(-(n_modes[1]//2), (n_modes[1]+1)//2)
    y, x = meshgrid(y, x, indexing="ij")
    k = stack((x.ravel(), y.ravel()), axis=1)
    return k

def nudft2d2(k:NDArray, f:NDArray, isign:Literal[1,-1]=-1) -> NDArray:
    """
    Type-2 (uniform to non-uniform) non-uniform discrete Fourier transform

    Args:
        k (NDArray): k-space coordinates, shape: (nK, 2).
        f (NDArray): image, shape: (nT,nY,nX) or (nY,nX).
        isign(Literal): sign in exponential.

    Returns:
        NDArray: NUDFT result, shape: (nK,)
    """
    kDst, f = asarray(k), asarray(f)
    if f.ndim==2: f = f[newaxis, ...]
    n_trans, n_modes = f.shape[0], f.shape[1:]

    kSrc = kCartesian(n_modes)
    f = f.reshape(n_trans, -1)

    out = _dft(f, kSrc, kDst, isign==1)
    return out


