import numpy as np
import pandas as pd
from astropy.table import Table
import astropy.units as u
import corner
import matplotlib.pyplot as plt

import torch
from sbi import utils as utils
from sbi.inference.base import infer

from interpolate_model_grid import InterpolateModelGrid
from galactic_extinction_correction import galactic_extinction_correction, rest_to_obs_wavelength

large_number=1e10


#import photoionization models
path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'
model_labels = list(np.load(path + 'test_model_high_res_age_2Myr_unattenuated_emission_line_labels.npy'))
model_flux = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_emission_line_luminosity_file.npy')
model_parameters = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_parameters_file.npy')
model_derived_parameters = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_derived_parameters_file.npy')

interpolated_grid = InterpolateModelGrid(model_labels, model_flux, model_parameters, model_derived_parameters, normalize_by='H__1_656281A')
interpolated_derived_parameters = interpolated_grid.interpolate_derived_parameters()

interpolated_F, interpolated_logOH= interpolated_derived_parameters[0], interpolated_derived_parameters[1]

line_labels = ['OII_3726', 'OII_3729', 'HBETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'HALPHA', 'NII_6584', 'SII_6716', 'SII_6731']
line_flux_labels = [label+'_FLUX' for label in line_labels]
line_flux_ivar_labels = [label+'_FLUX_IVAR' for label in line_labels]
line_wavelengths = [3727., 3729., 4862., 4960., 5008., 6549., 6564., 6585., 6718., 6732.]

denali_fastspec = Table.read("/Users/dirk/Documents/PhD/scripts/desi/data/Denali/fastspec-denali-cumulative.fits", hdu=1)
names = [name for name in denali_fastspec.colnames if len(denali_fastspec[name].shape) <= 1]
denali_fastspec = denali_fastspec[names].to_pandas()
denali_fastspec_hdu2 = Table.read("/Users/dirk/Documents/PhD/scripts/desi/data/Denali/fastspec-denali-cumulative.fits", hdu=2).to_pandas()

# masking operations
gal_mask = denali_fastspec_hdu2['SPECTYPE']==b'GALAXY'
sn_mask = denali_fastspec['HALPHA_FLUX']*(denali_fastspec['HALPHA_FLUX_IVAR']**0.5) > 25.0
z_mask = denali_fastspec['CONTINUUM_Z']<0.5
sf_mask = [np.log10(denali_fastspec["OIII_5007_FLUX"]/denali_fastspec["HBETA_FLUX"]) < 0.61*(np.log10(denali_fastspec["NII_6584_FLUX"]/denali_fastspec["HALPHA_FLUX"]) + 0.05)**-1 + 1.3][0]
line_num_mask = np.sum(denali_fastspec[line_flux_labels].to_numpy()!=0.0, axis=1)>=4.

denali_fastspec_hdu2 = denali_fastspec_hdu2[gal_mask & sn_mask & z_mask & sf_mask & line_num_mask].reset_index()
denali_fastspec = denali_fastspec[gal_mask & sn_mask & z_mask & sf_mask & line_num_mask].reset_index()

print(len(denali_fastspec))

#Extinction correction
extinction_correction_factor = np.ones((len(denali_fastspec), 10))

for i in range(len(denali_fastspec)):
    print('Extinction correction: ', i)
    obs_wavelength = rest_to_obs_wavelength(line_wavelengths * u.angstrom, denali_fastspec['CONTINUUM_Z'][i])
    extinction_correction_factor[i] = galactic_extinction_correction(denali_fastspec_hdu2['RA'][i]*u.degree, denali_fastspec_hdu2['DEC'][i]*u.degree, obs_wavelength, np.ones_like(line_wavelengths)*u.erg * u.cm**-2 * u.s**-1).value


#Fill analysis dataframe
data_df = pd.DataFrame()
data_df['TARGETID'] = denali_fastspec['TARGETID']


for i in range(len(line_labels)):
    print(extinction_correction_factor[i])
    data_df[line_labels[i]+'_FLUX'] = denali_fastspec[line_labels[i]+'_FLUX'] * extinction_correction_factor[:, i]
    data_df[line_labels[i]+'_FLUX_ERR'] = denali_fastspec[line_labels[i]+'_FLUX_IVAR']**-0.5 * extinction_correction_factor[:, i]

print(data_df)

def calc_log_dust(logtau):
    '''
    Calculate the dust surface density as in Brinchmann et al. 2013
    :param logtau: Log optical depth at 5500 Angstrom
    :return: Log of dust surface density
    '''
    return np.log10(0.2 * 10**logtau)

def calc_log_gas(logZ, xi, logtau):
    '''
    Calculate the gas surface density as in Brinchmann et al. 2013
    :param logZ: Log metallicity in units of solar metallicity
    :param xi: Dust-to-metal ratio
    :param logtau: Log optical depth at 5500 Angstrom
    :return: Log of gas surface density
    '''
    Zsun = 0.0142 # Asplund (2009) photospheric mass fraction
    return np.log10(0.2 * 10**logtau/(xi * 10**logZ * Zsun))

def prepare_input(flux, flux_error):
    '''
    Function to fill undetected emission lines with noise
    :param flux: emission line flux array
    :param flux_error: emission line flux error array
    :return:
    '''
    for i in range(len(flux)):
        if flux[i] == 0. or flux_error[i] == 0. or np.isinf(flux_error)[i]:
            flux_error[i] = 5 * np.max(flux)
            flux[i] = np.random.normal() * flux_error[i]
    output = np.expand_dims(np.concatenate([flux, flux_error]), axis=0)
    return torch.from_numpy(output)

#Model fitting function
def fit_model_to_df(index, prior_min=np.array([[-large_number, -1., -4., 0.1, -2.]]), prior_max=np.array([[large_number, 0.7, -1.0, 0.6, 0.6]]), plotting=False):
    '''
    Fitting SBI model to data.
    :param index: Unique index in the data_df
    :param prior_min: Prior lower bounds
    :param prior_max: Prior upper bounds
    :param plotting: Boolean to decide if corner plots should be generated.
    :return:
    '''
    parameters_out = np.ones((25)) * -999.
    galaxy = data_df[data_df['TARGETID'] == index]
    parameters_out[0] = galaxy['TARGETID'].to_numpy()[0]
    data_flux = galaxy[
        ['OII_3726_FLUX', 'OII_3729_FLUX', 'HBETA_FLUX', 'OIII_4959_FLUX', 'OIII_5007_FLUX', 'NII_6548_FLUX',
         'HALPHA_FLUX', 'NII_6584_FLUX', 'SII_6716_FLUX', 'SII_6731_FLUX']].to_numpy()[0]
    data_flux_error = galaxy[
        ['OII_3726_FLUX_ERR', 'OII_3729_FLUX_ERR', 'HBETA_FLUX_ERR', 'OIII_4959_FLUX_ERR', 'OIII_5007_FLUX_ERR',
         'NII_6548_FLUX_ERR', 'HALPHA_FLUX_ERR', 'NII_6584_FLUX_ERR', 'SII_6716_FLUX_ERR',
         'SII_6731_FLUX_ERR']].to_numpy()[0]

    posterior_samples = posterior.sample((10000,), x=infer_denali(data_flux,data_flux_error))
    posterior_samples = posterior_samples.numpy()

    sample_mask = np.prod((posterior_samples > prior_min)[:,1:] & (posterior_samples < prior_max)[:,1:], axis=1)==1
    masked_posterior_samples = posterior_samples[sample_mask]

    if np.sum(sample_mask)/len(sample_mask) > 0.5:
        samples_logOH = interpolated_logOH(masked_posterior_samples[:,1:-1])
        samples_dust  = calc_log_dust(masked_posterior_samples[:,4])
        samples_gas   = calc_log_gas(masked_posterior_samples[:,1],masked_posterior_samples[:,3],masked_posterior_samples[:,4])
        masked_posterior_samples = np.hstack([masked_posterior_samples, np.expand_dims(samples_logOH, axis=-1), np.expand_dims(samples_dust, axis=-1), np.expand_dims(samples_gas, axis=-1)])
        parameters_out[1:] = np.percentile(masked_posterior_samples, [16, 50, 84], axis=0).T.flatten()
        if plotting:
            corner.corner(masked_posterior_samples, show_titles=True, quantiles=[0.16, 0.5, 0.84], labels=['Amplitude', 'Z', 'U', '$\\xi$', '$\\tau$', 'log(O/H)', 'log(dust)', 'log(gas)'])
            plt.savefig('./plots/corner_{}.pdf'.format(int(parameters_out[0])))
            plt.close()
    return parameters_out

#Define number of dimensions and prior
num_dim = 5

prior = utils.BoxUniform(low = torch.tensor([-large_number, -large_number, -large_number, -large_number, -large_number]),
                         high= torch.tensor([large_number, large_number, large_number, large_number, large_number]))

#Create a fake simulation to instantiate a posterior
def fake_simulation(theta):
    return torch.tensor([[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]])

#Create posterior, do minimal simulations
posterior = infer(fake_simulation, prior, 'SNPE', num_simulations=10, )
#Replace posterior neural net with trained neural net from file

posterior.net = torch.load('./sbi_inference_larger_grid_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_train_285120/epoch_80'.format(i))
parameters = np.ones((len(data_df), 25)) * -999.
for j in range(len(data_df)):
    parameters[j] = fit_model_to_df(data_df['TARGETID'][j])
    if j % 100 == 0.0:
        np.save('sbi_fits_larger_grid_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_train_285120_epoch_80.npy'.format(i), parameters)

np.save('sbi_fits_larger_grid_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_train_285120_epoch_80.npy'.format(i), parameters)

