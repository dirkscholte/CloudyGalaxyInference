import numpy as np
import pandas as pd
from astropy.io import fits
import torch

import astropy.units as u
from galactic_extinction_correction import galactic_extinction_correction

from interpolate_model_grid import InterpolateModelGrid
from gaussian_noise_model import GaussianNoiseModelMUSE

from sbi import utils as utils
from sbi.inference import SNPE, simulate_for_sbi

# Inputs
redshift   = 0.005224
ra         = 185.4788625     # in degrees
dec        = 4.4737744       # in degrees
MUSE_cube  = fits.open("/Users/dirk/Documents/PhD/scripts/catalogs/MUSE/ADP.2021-07-16T10_20_56.494.fits")
model_path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'


# Train model for 'HBETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'HALPHA', 'NII_6584'

line_labels = ['H_BETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'H_ALPHA', 'NII_6584']
line_flux_labels = [label+'_FLUX' for label in line_labels]
line_flux_err_labels = [label+'_FLUX_ERR' for label in line_labels]
line_wavelengths = np.array([4862., 4960., 5008., 6549., 6564., 6585.])
obs_line_wavelength = line_wavelengths * (1 + redshift)

# Calculations of correction factors due to galactic foreground dust attenuation. Using the O'Donnell 94 correction scheme and SFD dust maps.
foreground_correction_factors = galactic_extinction_correction(ra*u.degree, dec*u.degree, line_wavelengths*u.angstrom, np.ones_like(line_wavelengths)*(u.erg * u.cm**-2 * u.s**-1)).value

cube_dims = MUSE_cube[6].data.shape
MUSE_df = pd.DataFrame()

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

MUSE_df=MUSE_df.fillna(0.0)

# Import photoionization models
model_labels = list(np.load(model_path + 'full_model_high_res_age_2Myr_unattenuated_emission_line_labels.npy'))
model_flux = np.load(model_path + 'full_model_high_res_age_2Myr_unattenuated_emission_line_luminosity_file.npy')
model_parameters = np.load(model_path + 'full_model_high_res_age_2Myr_unattenuated_parameters_file.npy')
model_derived_parameters = np.load(model_path + 'full_model_high_res_age_2Myr_unattenuated_derived_parameters_file.npy')

# Interpolate photoionization models
interpolated_grid = InterpolateModelGrid(model_labels, model_flux, model_parameters, model_derived_parameters, normalize_by='H__1_656281A')
interpolated_flux = interpolated_grid.interpolate_flux(['H__1_486133A', 'O__3_495891A', 'O__3_500684A', 'N__2_654800A', 'H__1_656281A', 'N__2_658345A'])

# Create gaussian noise model
gaussian_noise_model = GaussianNoiseModelMUSE(MUSE_df[line_flux_err_labels].to_numpy().reshape(-1), np.tile(obs_line_wavelength, (len(MUSE_df),1)).reshape(-1))

def transmission_function(lambda_, logtau, n=-1.3):
    '''
    Function to calculate the transmission function. As in Charlot and Fall (2000)
    :param lambda_: Wavelength values of the spectrum bins in Angstrom
    :param logtau: Log optical depth at 5500 Angstrom
    :param n: Exponent of power law. Default is -1.3 as is appropriate for birth clouds (-0.7 for general ISM).
    :return: Transmission function for each bin in the spectrum
    '''
    lambda_ = np.array(lambda_)
    return np.exp(-10**logtau * (lambda_/5500)**n)

def simulation(theta, redshift=redshift):
    '''
    Function to simulate emission line observations from photoionization models and a Gaussian noise model.
    :param theta: Input vector containing the free parameters of the model (Amplitude, Z, U, xi, tau)
    :param redshift: The redshift at which the observations are simulated.
    :return:
    '''
    theta = theta.numpy()[0]
    transmission = transmission_function(line_wavelengths, theta[-1])
    model_line_flux = np.zeros((len(interpolated_flux)))
    for i in range(len(interpolated_flux)):
        model_line_flux[i] = 10**theta[0] * interpolated_flux[i](theta[1:-1])*transmission[i]
    if redshift=='random':
        redshift = np.random.uniform(low=0.0, high=0.5)
    line_flux, line_flux_error = gaussian_noise_model.add_gaussian_noise(model_line_flux, (1 + redshift) * line_wavelengths, np.ones_like(line_wavelengths) * np.random.uniform(low=0.0, high=100.0))
    tensor_out = np.expand_dims(np.hstack([line_flux, line_flux_error]),axis=0)
    tensor_out = torch.from_numpy(tensor_out).to(torch.float32)
    return tensor_out

# Set SBI model number of free parameters and the prior parameter range for the free model parameters
num_dim = 5
prior = utils.BoxUniform(low = torch.tensor([0, -1., -4., 0.1, -2.]),
                         high= torch.tensor([6, 0.7, -1., 0.6, 0.6]))

if True:
    theta, x = simulate_for_sbi(simulation, proposal=prior, num_simulations=1000000)
    inference = SNPE(prior=prior)
    inference = inference.append_simulations(theta, x)

    save_epochs = np.arange(5,500,5)

    for epoch in save_epochs:
        if epoch==save_epochs[0]:
            density_estimator = inference.train(max_num_epochs=epoch, resume_training=False)
        else:
            density_estimator = inference.train(max_num_epochs=epoch, resume_training=True)
        posterior = inference.build_posterior(density_estimator)
        torch.save(posterior.net, './test_epoch_{}'.format(epoch))
        print('SAVED EPOCH: {}'.format(epoch))
        if inference._converged(epoch, 20):
            break
