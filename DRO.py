import numpy as np
import scipy.special as sf

def new():
    return DRO()

def kspace_to_image(kspace):
    """ Transform from k-space to image.
        This transform is normalized such that the image signal
        is the magnetization within one pixel, i.e. density x pixel_area.
    """
    img = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(kspace)))
    scale = 1/np.prod(kspace.shape)
    return img*scale

def image_to_kspace(image):
    """ Transform from image to k-space.
        The transform is normalized such that it is the inverse of
        kspace_to_image.
        I.e. the following identity holds:
        kspace = image_to_kspace(kspace_to_image(kspace))
    """
    kspace = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(image)))
    scale = np.prod(kspace.shape)
    return kspace*scale

def _kspace_grid(fov, matrix):
    """ Compute values of kx and ky for given field of view and matrix.
        I.e. these are the coordinates of the discretized k-space.
    """
    n = matrix
    dk = 1/fov

    kx = dk[0]*(np.arange(n[0]) - n[0]//2)
    ky = dk[1]*(np.arange(n[1]) - n[1]//2)
    return kx, ky

def _shift(kspace, kx, ky, shift):
    """ Apply phase ramp to k-space which is equivalent to shift
        in image space.
    """
    xphase = np.exp(1j*(2*np.pi*shift[0])*kx, dtype=np.complex64)
    yphase = np.exp(1j*(2*np.pi*shift[1])*ky, dtype=np.complex64)
    kspace *= xphase[np.newaxis, :]
    kspace *= yphase[:, np.newaxis]

# DRO is a container for geometrical primitives.
class DRO:
    def __init__(self):
        self.elements = []

    def add(self, e):
        """ Add an object to the DRO

        Parameters:
        -----------
        e : object
            Can be a DRO or a geometrical primitive (Disk, Rectangle, Gaussian)
        """

        if type(e) == DRO:
            self.elements += e.elements
        elif type(e) in [Disk, Rectangle, Gaussian]:
            self.elements.append(e)
        else:
            raise ValueError("Invalid object type: {}".format(type(e)))

    def image(self, fov, matrix, ovs):
        """ Compute image by sampling the DRO in image space.

        Parameters:
        -----------
        fov : np.array with length 2
            Field of view of the generated image in mm.
            (fov_x, fov_y) = fov

        matrix : np.array with length 2, dtype=int
            Number of pixels in both directions.
            (nx, ny) = matrix

        Returns:
        --------
        img : 2D np.array, dtype=complex64
            Image of the discretized DRO. Note that the x-dimension is last
            in the image array, but is first in fov and matrix parameters.
            img.shape = (ny, nx)
        """
        if ((type(ovs) != int) or (ovs < 1)):
            raise ValueError("ovs must be positive, odd integer.")

        (nx, ny) = matrix
        img = np.zeros((ny, nx), np.complex64)
        for obj in self.elements:
            img += obj.image(fov, matrix, ovs)
        return img

    def kspace(self, fov, matrix):
        """ Compute k-space by sampling the Fourier-transform of the DRO.

        Parameters:
        -----------
        fov : np.array with length 2
            Field of view of the generated k-space in mm.
            (fov_x, fov_y) = fov

        matrix : np.array with length 2, dtype=int
            Number of pixels in both directions.
            (nx, ny) = matrix

        Returns:
        --------
        kspace : 2D np.array, dtype=complex64
            Discretized k-space of the DRO. Note that the kx-dimension is last
            in the kspace array, but is first in fov and matrix parameters.
            kspace.shape = (ny, nx)
        """
        (nx, ny) = matrix
        signal = np.zeros((ny, nx), np.complex64)
        for obj in self.elements:
            signal += obj.kspace(fov, matrix)
        return signal

""" Definition of the geometrical primitives which can be added to a DRO.
    Each object must implement the functions:
        kspace(self, fov, matrix)

    The scaling of the objects can be controlled by setting one of the two
    parameters:
    1. magnetization: this is defined as the value of the Fourier integral of
    the object at k==0. I.e. it is the total magnetization of the object.
    2. density: this is the magnetization of the object per square-millimeter.
"""

class Disk:
    def __init__(self, center, radius, magnetization=None, density=None):
        """ Disk.
        The scaling can be controlled by setting one of the two parameters:
        1. magnetization:
        This is defined as the value of the Fourier integral of
        the object at k==0. I.e. it is the total magnetization of the object.
        2. density:
        This is the magnetization of the object per square-millimeter.

        Parameters:
        -----------
        center : 2 values
            x,y coordinates of the center of the disk in mm.
        radius : float
            radius of the disk in mm.
        magnetization or density : complex
            Scaling (see above).
        """

        if magnetization is not None:
            if density is not None:
                raise ValueError("Density and magnetization parameters"
                                 " are mutually exclusive.")
            if np.iscomplex(magnetization):
                self.m = magnetization
            else:
                self.m = magnetization + 0.0j
            self.density = self.m / (np.pi * radius**2)
        else:
            if density is None:
                raise ValueError("Either density or magnetization parameter"
                                 " must be set.")
            if np.iscomplex(density):
                self.density = density
            else:
                self.density = density + 0.0j
            self.m = self.density * (np.pi * radius**2)

        if type(center) != np.array:
            self.center = np.array(center, np.float32)
        else:
            self.center = np.copy(center)
        self.radius = radius

    def kspace(self, fov, matrix):
        """ Compute discrete k-space for given field-of-view and matrix.
            The k-space is scaled such that the amplitude in the k-space
            center is equal to the magnetization of the disk (self.m)
        """
        kx, ky = _kspace_grid(fov, matrix)
        Kx, Ky = np.meshgrid(kx, ky)
        kr = np.sqrt(Kx**2+Ky**2, dtype=np.float32)
        z = (2*np.pi*self.radius) * kr
        ik0 = matrix//2  # index of k-space center
        z[ik0[1], ik0[0]] = 1E-4
        # this avoids division by zero problems below
        # the relative error introduced by this is ~1E-9
        # which is below machine precision for floats
        scale = 2*self.m  # limit of jv(z)/z for z->0 is 0.5
        kspace = scale * (sf.jv(1, z)/z)
        _shift(kspace, kx, ky, self.center)
        return kspace

class Rectangle:
    def __init__(self, center, size, rotation, magnetization=None,
                 density=None):
        """ Rectangle.
        The scaling can be controlled by setting one of the two parameters:
        1. magnetization:
        This is defined as the value of the Fourier integral of
        the object at k==0. I.e. it is the total magnetization of the object.
        2. density:
        This is the magnetization of the object per square-millimeter.

        Parameters:
        -----------
        center : 2 float values
            x,y coordinates of the center of the rectangle in mm.
        size : 2 float values
            size in x and y direction in mm.
        rotation : float
            Rotation angle about the center of the rectangle in rad.
        magnetization or density : complex
            Scaling (see above).
        """
        if magnetization is not None:
            if density is not None:
                raise ValueError("Density and magnetization parameters"
                                 " are mutually exclusive.")
            if np.iscomplex(magnetization):
                self.m = magnetization
            else:
                self.m = magnetization + 0.0j
            self.density = self.m / np.prod(size)
        else:
            if density is None:
                raise ValueError("Either density or magnetization parameter"
                                 " must be set.")
            if np.iscomplex(density):
                self.density = density
            else:
                self.density = density + 0.0j
            self.m = self.density * np.prod(size)

        if type(center) != np.array:
            self.center = np.array(center, np.float32)
        else:
            self.center = np.copy(center)

        if type(size) != np.array:
            self.size = np.array(size, np.float32)
        else:
            self.size = np.copy(size)
        self.rotation = rotation

    def kspace(self, fov, matrix):
        """ Compute discrete k-space for given field-of-view and matrix.
            The k-space is scaled such that the amplitude in the k-space
            center is equal to the magnetization (self.m)
        """
        kx, ky = _kspace_grid(fov, matrix)
        Kx, Ky = np.meshgrid(kx, ky)
        Kxr = np.cos(self.rotation) * Kx - np.sin(self.rotation) * Ky
        Kyr = np.sin(self.rotation) * Kx + np.cos(self.rotation) * Ky
        scale = self.m
        kspace = scale * (np.sinc(Kxr*self.size[0])
                          * np.sinc(Kyr*self.size[1]))
        _shift(kspace, kx, ky, self.center)
        return kspace

class Gaussian:
    def __init__(self, center, sigma, magnetization=None, density=None):
        """ Magnetization varies with distance from center according to
        Gaussian profile.
        The scaling can be controlled by setting one of the two parameters:
        1. magnetization:
        This is defined as the value of the Fourier integral of
        the object at k==0. I.e. it is the total magnetization of the object.
        2. density:
        This is the magnetization of the object per square-millimeter.

        Parameters:
        -----------
        center : 2 values
            x,y coordinates of the center of the Gaussian in mm.
        sigma : float
            Standard deviation of the Gaussian in mm.
        magnetization or density : complex
            Scaling (see above).
        """
        if magnetization is not None:
            if density is not None:
                raise ValueError("Density and magnetization parameters"
                                 " are mutually exclusive.")
            if np.iscomplex(magnetization):
                self.m = magnetization
            else:
                self.m = magnetization + 0.0j
            self.density = self.m / (2*np.pi * sigma**2)
        else:
            if density is None:
                raise ValueError("Either density or magnetization parameter"
                                 " must be set.")
            if np.iscomplex(density):
                self.density = density
            else:
                self.density = density + 0.0j
            self.m = self.density * (2*np.pi * sigma**2)

        if type(center) != np.array:
            self.center = np.array(center, np.float32)
        else:
            self.center = np.copy(center)
        self.sigma = sigma

    def kspace(self, fov, matrix):
        """ Compute discrete k-space for given field-of-view and matrix.
            The k-space is scaled such that the amplitude in the k-space
            center is equal to the magnetization of the disk (self.m)
        """
        kx, ky = _kspace_grid(fov, matrix)
        Kx, Ky = np.meshgrid(kx, ky)
        kr2 = Kx**2 + Ky**2
        z = (2*(np.pi*self.sigma)**2) * kr2
        scale = self.m
        kspace = scale * np.exp(-z)
        _shift(kspace, kx, ky, self.center)
        return kspace