import numpy as np
import h5py
from astropy.io import ascii, fits
import astropy.units as u
from astropy.constants import c
speed_of_light = c.to(u.km/u.s).value

def doppler(wave, v):
    """
    Applies a Doppler shift of velocity v to wavelength grid.
    Relativity-safe.

    inputs
    ------
    wave : float or array, wavelength
    v : float, velocity in km/s

    output
    ------
    new_wave : `wave` shifted to a new rest frame at velocity `v`
    (also a solid genre of music)
    """
    frac = (1. - v/speed_of_light) / (1. + v/speed_of_light)
    return wave * np.sqrt(frac)

def read_spectrum(filename, bc=0.):
    """
    Read in a spectrum, barycentric correct it, and do a rough normalization.

    inputs
    ------
    filename : string, name of the file
    bc : float, optional; barycentric correction factor in km/s

    outputs
    -------
    w : (n_order, n_pixel) grid of barycentric-corrected wavelengths
    f : roughly continuum normalized fluxes on same grid as w
    s : estimated uncertainties for f
    """
    with fits.open(filename) as hdus:
        f = np.copy(hdus[0].data) # unnormalized fluxes
        s = np.sqrt(f) # uncertainty on f
        w = np.copy(hdus[2].data) # obs rest frame wavelengths
        w = doppler(w, bc) # barycentric correct
        for j in range(np.shape(f)[0]): # normalize each order
            upper_cut = np.nanmedian(f[j,:]) + np.std(f[j,:]) * 3
            bad_pix = (f[j,:] < 0) | (f[j,:] >= upper_cut)
            f[j,bad_pix] = np.nan
            norm_factor = np.nanmedian(f[j,:])
            f[j,:] = f[j,:]/norm_factor
            s[j,:] = s[j,:]/norm_factor
    return w,f,s

if __name__ == '__main__':
    spectra_dir = '/Users/mbedell/python/PSOAP/data/giraffe_hires/'
    metadata_file = '/Users/mbedell/python/PSOAP/data/tic102780470_trv_hires.csv'

    # Load/Create an HDF5 file
    f = h5py.File('/Users/mbedell/python/PSOAP/data/giraffe_hires.hdf5','a')

    # Load in the dates
    metadata = np.genfromtxt(metadata_file, skip_header=8, 
                            delimiter=',',dtype=None,encoding=None, 
                            names=True)
    f["JD"] = metadata['bjd']

    # Load in the spectra
    n_epochs = len(metadata)
    n_orders = 23 + 16 + 10 # blue + red + IR
    n_pix = 4021
    flux = np.zeros((n_epochs, n_orders, n_pix))
    wave = np.zeros_like(flux)
    sigma = np.zeros_like(flux)
    for i in range(n_epochs):
        # read the blue chip:
        w,fl,s = read_spectrum(spectra_dir+'b{0}.fits'.format(metadata['observation_id'][i]), bc=metadata['bc'][i]*-1e-3)
        flux[i,0:23,:] = np.copy(fl)
        wave[i,0:23,:] = np.copy(w)
        sigma[i,0:23,:] = np.copy(s)
        # read the red chip:
        w,fl,s = read_spectrum(spectra_dir+'r{0}.fits'.format(metadata['observation_id'][i]), bc=metadata['bc'][i]*-1e-3)
        flux[i,23:39,:] = np.copy(fl)
        wave[i,23:39,:] = np.copy(w)
        sigma[i,23:39,:] = np.copy(s)
        # read the IR chip:
        w,fl,s = read_spectrum(spectra_dir+'i{0}.fits'.format(metadata['observation_id'][i]), bc=metadata['bc'][i]*-1e-3)
        flux[i,39:,:] = np.copy(fl)
        wave[i,39:,:] = np.copy(w)
        sigma[i,39:,:] = np.copy(s)

    f["wl"] = wave
    f["fl"] = flux
    f["sigma"] = sigma
    f["BCV"] = metadata['bc']*1.e-3

    # Now save this
    f.close()
