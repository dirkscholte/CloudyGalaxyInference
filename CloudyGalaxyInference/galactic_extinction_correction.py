import astropy.units as u
from dust_extinction.parameter_averages import O94
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord

sfd = SFDQuery()
ext = O94(Rv=3.1)

@u.quantity_input
def galactic_extinction_correction(ra: u.degree, dec: u.degree, obs_wavelengths: u.angstrom, spectrum: u.erg * u.cm**-2 * u.s**-1):
    coords = SkyCoord(ra=ra, dec=dec, frame='icrs')
    ebv = sfd(coords)
    spectrum_noext = spectrum/ext.extinguish(obs_wavelengths, Ebv=ebv)
    return spectrum_noext

@u.quantity_input
def rest_to_obs_wavelength(rest_wavelength: u.angstrom, redshift):
    return rest_wavelength * (1 + redshift)
