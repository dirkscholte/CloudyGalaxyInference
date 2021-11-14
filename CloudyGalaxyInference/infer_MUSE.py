import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
import corner
import matplotlib.pyplot as plt

import torch
from sbi import utils as utils
from sbi.inference.base import infer

from interpolate_model_grid import InterpolateModelGrid
from galactic_extinction_correction import galactic_extinction_correction, rest_to_obs_wavelength

large_number = 1e10

# Inputs
redshift   = 0.005224
ra         = 185.4788625     # in degrees
dec        = 4.4737744       # in degrees
MUSE_cube  = fits.open("/Users/dirk/Documents/PhD/scripts/catalogs/MUSE/ADP.2021-07-16T10_20_56.494.fits")
model_path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'
posterior_network = 'sbi_inference_MUSE_train_1M_Hb_OIII_OIII_NII_Ha_NII_epoch_35'
output_file = 'sbi_fits_MUSE_train_1M_Hb_OIII_OIII_NII_Ha_NII_epoch_35.npy'

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

line_labels = ['H_BETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'H_ALPHA', 'NII_6584']
line_flux_labels = [label+'_FLUX' for label in line_labels]
line_flux_err_labels = [label+'_FLUX_ERR' for label in line_labels]
line_wavelengths = [4862., 4960., 5008., 6549., 6564., 6585.]

foreground_correction_factors = galactic_extinction_correction(ra*u.degree, dec*u.degree, line_wavelengths*u.angstrom, np.ones_like(line_wavelengths)*(u.erg * u.cm**-2 * u.s**-1)).value

cube_dims = MUSE_cube[6].data.shape
MUSE_df = pd.DataFrame()

MUSE_df['SPAXELID'] = np.arange(len(MUSE_cube[6].data.reshape(-1)))
MUSE_df['INDEX_I'] = np.meshgrid(np.arange(cube_dims[0]),np.arange(cube_dims[1]))[0].reshape(-1)
MUSE_df['INDEX_J'] = np.meshgrid(np.arange(cube_dims[0]),np.arange(cube_dims[1]))[1].reshape(-1)

MUSE_df['H_BETA_FLUX'] = MUSE_cube[6].data.reshape(-1) * foreground_correction_factors[0]
MUSE_df['H_BETA_FLUX_ERR'] = MUSE_cube[7].data.reshape(-1) * foreground_correction_factors[0]

MUSE_df['OIII_4959_FLUX'] = MUSE_cube[12].data.reshape(-1) * foreground_correction_factors[1]
MUSE_df['OIII_4959_FLUX_ERR'] = MUSE_cube[13].data.reshape(-1) * foreground_correction_factors[1]

MUSE_df['OIII_5007_FLUX'] = MUSE_cube[18].data.reshape(-1) * foreground_correction_factors[2]
MUSE_df['OIII_5007_FLUX_ERR'] = MUSE_cube[19].data.reshape(-1) * foreground_correction_factors[2]

MUSE_df['NII_6548_FLUX'] = MUSE_cube[24].data.reshape(-1) * foreground_correction_factors[3]
MUSE_df['NII_6548_FLUX_ERR'] = MUSE_cube[25].data.reshape(-1) * foreground_correction_factors[3]

MUSE_df['H_ALPHA_FLUX'] = MUSE_cube[30].data.reshape(-1) * foreground_correction_factors[4]
MUSE_df['H_ALPHA_FLUX_ERR'] = MUSE_cube[31].data.reshape(-1) * foreground_correction_factors[4]

MUSE_df['NII_6584_FLUX'] = MUSE_cube[36].data.reshape(-1) * foreground_correction_factors[5]
MUSE_df['NII_6584_FLUX_ERR'] = MUSE_cube[37].data.reshape(-1) * foreground_correction_factors[5]

MUSE_df = MUSE_df.fillna(0.0)

# Fill analysis dataframe
data_df = MUSE_df

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

def prepare_input(flux, flux_error):
    '''
    fill unrealistic values with 0.0
    :param flux:
    :param flux_error:
    :return:
    '''
    for i in range(len(flux)):
        if flux_error[i] <= 0. or np.isinf(flux_error)[i]:
            flux_error[i] = 0.0
            flux[i] = 0.0
    output = np.expand_dims(np.concatenate([flux, flux_error]), axis=0)
    return torch.from_numpy(output)

# Model fitting function
def fit_model_to_df(index, prior_min=np.array([[0., -1., -4., 0.1, -2.]]),
                    prior_max=np.array([[5., 0.7, -1., 0.6, 0.6]]), plotting=False):
    '''
    Function to transform fitting output to parameter percentile values in a numpy array.
    :param index:
    :param prior_min:
    :param prior_max:
    :param plotting:
    :return:
    '''
    parameters_out = np.ones((25)) * -999.
    galaxy = data_df[data_df['SPAXELID'] == index]
    parameters_out[0] = galaxy['SPAXELID'].to_numpy()[0]
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
    return torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])


# Create posterior, do minimal simulations
posterior = infer(fake_simulation, prior, 'SNPE', num_simulations=10, )
# Replace posterior neural net with trained neural net from file
posterior.net = torch.load(posterior_network)
parameters = np.ones((len(data_df), 25)) * -999.
parameters[:,0] = data_df['SPAXELID'].to_numpy()
for j in range(len(data_df)):
    parameters[j] = fit_model_to_df(data_df['SPAXELID'].iloc[j], plotting=False)
    if j % 100 == 0.0:
        np.save(output_file, parameters)

np.save(output_file, parameters)
