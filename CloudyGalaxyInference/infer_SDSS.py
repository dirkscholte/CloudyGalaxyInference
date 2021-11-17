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

large_number = 1e10

# import photoionization models
path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'
model_labels = list(np.load(path + 'test_model_high_res_age_2Myr_unattenuated_emission_line_labels.npy'))
model_flux = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_emission_line_luminosity_file.npy')
model_parameters = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_parameters_file.npy')
model_derived_parameters = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_derived_parameters_file.npy')

interpolated_grid = InterpolateModelGrid(model_labels, model_flux, model_parameters, model_derived_parameters,
                                         normalize_by='H__1_656281A')
interpolated_derived_parameters = interpolated_grid.interpolate_derived_parameters()

interpolated_F, interpolated_logOH = interpolated_derived_parameters[0], interpolated_derived_parameters[1]

line_labels = ['OII_3726', 'OII_3729', 'H_BETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'H_ALPHA', 'NII_6584']
line_flux_labels = [label+'_FLUX' for label in line_labels]
line_flux_err_labels = [label+'_FLUX_ERR' for label in line_labels]
line_wavelengths = [3727., 3729., 4862., 4960., 5008., 6549., 6564., 6585.]

sdss_spec_info = Table.read("/Users/dirk/Documents/PhD/scripts/catalogs/galSpecInfo-dr8.fits")
names = [name for name in sdss_spec_info.colnames if len(sdss_spec_info[name].shape) <= 1]
sdss_spec_info = sdss_spec_info[names].to_pandas()
sdss_spec_line = Table.read("/Users/dirk/Documents/PhD/scripts/catalogs/galSpecLine-dr8.fits").to_pandas()
sdss_spec_extra = Table.read("/Users/dirk/Documents/PhD/scripts/catalogs/galSpecExtra-dr8.fits").to_pandas()

print(sdss_spec_info.columns)
print(sdss_spec_line.columns)
print(sdss_spec_extra.columns)

print(sdss_spec_info['TARGETTYPE'])


# masking operations
gal_mask = sdss_spec_info['TARGETTYPE']==b'GALAXY             '
sn_mask = sdss_spec_line['H_ALPHA_FLUX']/sdss_spec_line['H_ALPHA_FLUX_ERR'] > 15.0
z_mask = (sdss_spec_info['Z']>0.027) & (sdss_spec_info['Z']<0.5)
sf_mask = ~[np.log10(sdss_spec_line["OIII_5007_FLUX"]/sdss_spec_line["H_BETA_FLUX"]) > 0.61*(np.log10(sdss_spec_line["NII_6584_FLUX"]/sdss_spec_line["H_ALPHA_FLUX"]) - 0.05)**-1 + 1.3][0]

sdss_spec_info = sdss_spec_info[gal_mask & sn_mask & z_mask & sf_mask]
sdss_spec_line = sdss_spec_line[gal_mask & sn_mask & z_mask & sf_mask]
sdss_spec_extra = sdss_spec_extra[gal_mask & sn_mask & z_mask & sf_mask]

# Fill analysis dataframe
data_df = pd.DataFrame()
data_df['TARGETID'] = sdss_spec_info['SPECOBJID']
data_df[line_flux_labels] = sdss_spec_line[line_flux_labels]
data_df[line_flux_err_labels] = sdss_spec_line[line_flux_err_labels]

def calc_log_dust(logtau):
    '''
    Calculate the dust surface density as in Brinchmann et al. 2013
    :param logtau: Log optical depth at 5500 Angstrom
    :return: Log of dust surface density
    '''
    return np.log10(0.2 * 10 ** logtau)


def calc_log_gas(logZ, xi, logtau):
    '''
    Calculate the gas surface density as in Brinchmann et al. 2013
    :param logZ: Log metallicity in units of solar metallicity
    :param xi: Dust-to-metal ratio
    :param logtau: Log optical depth at 5500 Angstrom
    :return: Log of gas surface density
    '''
    Zsun = 0.0142  # Asplund (2009) photospheric mass fraction
    return np.log10(0.2 * 10 ** logtau / (xi * 10 ** logZ * Zsun))


# Fill undetected emission lines with noise
def prepare_input(flux, flux_error):
    for i in range(len(flux)):
        if flux_error[i] <= 0. or np.isinf(flux_error)[i]:
            flux_error[i] = 0.0
            flux[i] = 0.0
    output = np.expand_dims(np.concatenate([flux, flux_error]), axis=0)
    return torch.from_numpy(output)


# Model fitting function
def fit_model_to_df(index, prior_min=np.array([[0., -1., -4., 0.1, -2.]]),
                    prior_max=np.array([[5., 0.7, -1., 0.6, 0.6]]), plotting=False):
    parameters_out = np.ones((25)) * -999.
    galaxy = data_df[data_df['TARGETID'] == index]
    parameters_out[0] = galaxy['TARGETID'].to_numpy()[0]
    data_flux = galaxy[line_flux_labels].to_numpy()[0]
    data_flux_error = galaxy[line_flux_err_labels].to_numpy()[0]

    posterior_samples = posterior.sample((10000,), x=prepare_input(data_flux, data_flux_error))
    posterior_samples = posterior_samples.numpy()

    sample_mask = np.prod((posterior_samples > prior_min)[:, 1:] & (posterior_samples < prior_max)[:, 1:], axis=1) == 1
    masked_posterior_samples = posterior_samples[sample_mask]

    if np.sum(sample_mask) / len(sample_mask) > 0.5:
        samples_logOH = interpolated_logOH(masked_posterior_samples[:, 1:-1])
        samples_dust = calc_log_dust(masked_posterior_samples[:, 4])
        samples_gas = calc_log_gas(masked_posterior_samples[:, 1], masked_posterior_samples[:, 3],
                                   masked_posterior_samples[:, 4])
        masked_posterior_samples = np.hstack(
            [masked_posterior_samples, np.expand_dims(samples_logOH, axis=-1), np.expand_dims(samples_dust, axis=-1),
             np.expand_dims(samples_gas, axis=-1)])
        parameters_out[1:] = np.percentile(masked_posterior_samples, [16, 50, 84], axis=0).T.flatten()
        if plotting:
            corner.corner(masked_posterior_samples, show_titles=True, quantiles=[0.16, 0.5, 0.84],
                          labels=['Amplitude', 'Z', 'U', '$\\xi$', '$\\tau$', 'log(O/H)', 'log(dust)', 'log(gas)'])
            plt.savefig('./plots/corner_{}.pdf'.format(int(parameters_out[0])))
            plt.close()
    return parameters_out


# Define number of dimensions and prior
num_dim = 5

prior = utils.BoxUniform(low=torch.tensor([-large_number, -large_number, -large_number, -large_number, -large_number]),
                         high=torch.tensor([large_number, large_number, large_number, large_number, large_number]))


# Create a fake simulation to instantiate a posterior
def fake_simulation(theta):
    return torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])


# Create posterior, do minimal simulations
posterior = infer(fake_simulation, prior, 'SNPE', num_simulations=10, )
# Replace posterior neural net with trained neural net from file
posterior.net = torch.load('sbi_inference_SDSS_train_5M_OII_OII_Hb_OIII_OIII_NII_Ha_NII_epoch_55')
parameters = np.ones((len(data_df), 25)) * -999.
parameters[:,0] = data_df['TARGETID'].to_numpy()
for j in range(len(data_df)):
    parameters[j] = fit_model_to_df(data_df['TARGETID'].iloc[j], plotting=False)
    if j % 100 == 0.0:
        np.save('sbi_fits_SDSS_train_5M_OII_OII_Hb_OIII_OIII_NII_Ha_NII_epoch_55.npy', parameters)

np.save('sbi_fits_SDSS_train_5M_OII_OII_Hb_OIII_OIII_NII_Ha_NII_epoch_55.npy', parameters)
