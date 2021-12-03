import pandas as pd
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

def galactic_extinction_correct_MUSE_df(dataframe, column_rest_wls, column_labels, galaxy_id_column='NAME'):
    corr_dataframe = pd.DataFrame()
    galaxy_ids = dataframe[galaxy_id_column].unique()
    for galaxy_id in galaxy_ids:
        sub_dataframe = dataframe[dataframe['NAME']==galaxy_id]
        ra = sub_dataframe['RA'].iloc[0]*u.degree
        dec = sub_dataframe['DEC'].iloc[0]*u.degree
        column_obs_wls = column_rest_wls * (1 + sub_dataframe['REDSHIFT'].iloc[0]) * u.angstrom
        galactic_correction_factors = galactic_extinction_correction(ra, dec, column_obs_wls, np.ones_like(column_rest_wls)*(u.erg * u.cm**-2 * u.s**-1)).value
        sub_dataframe[column_labels] *= galactic_correction_factors
        corr_dataframe = pd.concat([corr_dataframe, sub_dataframe])
    return corr_dataframe