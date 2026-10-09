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

