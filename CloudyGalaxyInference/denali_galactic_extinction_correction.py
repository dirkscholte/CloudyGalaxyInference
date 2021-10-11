import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from galactic_extinction_correction import galactic_extinction_correction, rest_to_obs_wavelength

hdul = fits.open("/Users/dirk/Documents/PhD/scripts/desi/data/Denali/fastspec-denali-cumulative.fits")
hdu1 = Table.read(hdul[1])
hdu2 = Table.read(hdul[2])

args = np.argwhere(np.array([colname[-5:] for colname in hdu1.colnames]) == '_FLUX').flatten()
print(args.dtype)
line_flux_labels = np.array(hdu1.colnames)[args]
line_labels = np.array([label[:-5] for label in line_flux_labels])
line_flux_ivar_labels = np.array([label+'_FLUX_IVAR' for label in line_labels])
line_wavelengths = np.array([2800., 3346., 3426., 3726., 3729., 3869., 4959., 5007., 3970., 4102., 4340., 4861., 6563., 6548., 6584., 6716., 6731., 9069., 9532.])


hdu3 = hdu1.copy()[list(np.concatenate([['TARGETID'], line_flux_labels, line_flux_ivar_labels]))]
names = [name for name in hdu1.colnames if len(hdu1[name].shape) <= 1]
hdu1_df = hdu1[names].to_pandas()
hdu3_df = hdu3.to_pandas()

for i in range(len(hdu3)):
    print('Foreground correction: ', i)
    obs_wavelength = rest_to_obs_wavelength(line_wavelengths * u.angstrom, hdu1_df['CONTINUUM_Z'][i])
    wavelength_mask = ((1/obs_wavelength.to(u.micron)).value > 0.3) & ((1/obs_wavelength.to(u.micron)).value < 10.0)

    extinction_correction_factor = galactic_extinction_correction(hdu2['RA'][i]*u.degree, hdu2['DEC'][i]*u.degree, obs_wavelength[wavelength_mask], np.ones_like(line_wavelengths[wavelength_mask])*u.erg * u.cm**-2 * u.s**-1).value
    hdu3[list(line_flux_labels[wavelength_mask])][i] = hdu1_df.iloc[i][line_flux_labels[wavelength_mask]] * extinction_correction_factor
    hdu3[list(line_flux_ivar_labels[wavelength_mask])][i] = hdu1_df.iloc[i][line_flux_ivar_labels[wavelength_mask]] * extinction_correction_factor**-2.
    if np.sum(~wavelength_mask)>0:
        hdu3[list(line_flux_labels[~wavelength_mask])][i] = np.ones((np.sum(~wavelength_mask))) * -999.
        hdu3[list(line_flux_ivar_labels[~wavelength_mask])][i] = np.ones((np.sum(~wavelength_mask))) * -999.




hdu3 = fits.table_to_hdu(hdu3)
hdu3.name = 'GAL_CORR'
hdul.append(hdu3)

hdul.info()
hdul.writeto('/Users/dirk/Documents/PhD/scripts/desi/data/Denali/fastspec-denali-cumulative-foreground-corr.fits', overwrite=True)
