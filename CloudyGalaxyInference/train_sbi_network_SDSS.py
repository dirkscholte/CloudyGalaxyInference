import numpy as np
import pandas as pd
from astropy.table import Table
import torch

import astropy.units as u
from galactic_extinction_correction import rest_to_obs_wavelength

from interpolate_model_grid import InterpolateModelGrid
from gaussian_noise_model import GaussianNoiseModelWavelength

from sbi import utils as utils
from sbi.inference import SNPE, simulate_for_sbi


# Train model for 'OII_3726', 'OII_3729', 'HBETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'HALPHA', 'NII_6584', 'SII_6716', 'SII_6731'

#import photoionization models
path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'
model_labels = list(np.load(path + 'full_model_high_res_age_2Myr_unattenuated_emission_line_labels.npy'))
model_flux = np.load(path + 'full_model_high_res_age_2Myr_unattenuated_emission_line_luminosity_file.npy')
model_parameters = np.load(path + 'full_model_high_res_age_2Myr_unattenuated_parameters_file.npy')
model_derived_parameters = np.load(path + 'full_model_high_res_age_2Myr_unattenuated_derived_parameters_file.npy')

interpolated_grid = InterpolateModelGrid(model_labels, model_flux, model_parameters, model_derived_parameters, normalize_by='H__1_656281A')
interpolated_flux = interpolated_grid.interpolate_flux(['O__2_372603A', 'O__2_372881A', 'H__1_486133A', 'O__3_495891A', 'O__3_500684A', 'N__2_654800A', 'H__1_656281A', 'N__2_658345A'])

line_labels = ['OII_3726', 'OII_3729', 'H_BETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'H_ALPHA', 'NII_6584']
line_flux_labels = [label+'_FLUX' for label in line_labels]
line_flux_err_labels = [label+'_FLUX_ERR' for label in line_labels]
line_wavelengths = np.array([3726., 3729., 4862., 4960., 5008., 6549., 6564., 6585.])

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
sn_mask = sdss_spec_line['H_ALPHA_FLUX']/sdss_spec_line['H_ALPHA_FLUX_ERR'] > 5.0
z_mask = (sdss_spec_info['Z']>0.027) & (sdss_spec_info['Z']<0.5)
sf_mask = ~[np.log10(sdss_spec_line["OIII_5007_FLUX"]/sdss_spec_line["H_BETA_FLUX"]) > 0.61*(np.log10(sdss_spec_line["NII_6584_FLUX"]/sdss_spec_line["H_ALPHA_FLUX"]) - 0.05)**-1 + 1.3][0]

sdss_spec_info = sdss_spec_info[gal_mask & sn_mask & z_mask & sf_mask]
sdss_spec_line = sdss_spec_line[gal_mask & sn_mask & z_mask & sf_mask]
sdss_spec_extra = sdss_spec_extra[gal_mask & sn_mask & z_mask & sf_mask]

obs_wavelength = np.zeros((len(sdss_spec_line), len(line_wavelengths)))
for i in range(len(line_wavelengths)):
    obs_wavelength[:,i] = line_wavelengths[i] * (1 + sdss_spec_info["Z"])

gaussian_noise_model = GaussianNoiseModelWavelength(sdss_spec_line[line_flux_err_labels].to_numpy().reshape(-1), obs_wavelength.reshape(-1))
gaussian_noise_model.plot_noise()


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

def simulation(theta, redshift='random'):
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

num_dim = 5
prior = utils.BoxUniform(low = torch.tensor([0, -1., -4., 0.1, -2.]),
                         high= torch.tensor([5, 0.7, -1., 0.6, 0.6]))

if True:
    theta, x = simulate_for_sbi(simulation, proposal=prior, num_simulations=5000000)
    inference = SNPE(prior=prior)
    inference = inference.append_simulations(theta, x)

    save_epochs = np.arange(5,500,5)

    for epoch in save_epochs:
        if epoch==save_epochs[0]:
            density_estimator = inference.train(max_num_epochs=epoch, resume_training=False)  # Pick `max_num_epochs` such that it does not exceed the runtime.
        else:
            density_estimator = inference.train(max_num_epochs=epoch, resume_training=True)  # Pick `max_num_epochs` such that it does not exceed the runtime.
        posterior = inference.build_posterior(density_estimator)
        torch.save(posterior.net, './sbi_inference_SDSS_train_5M_v3_OII_OII_Hb_OIII_OIII_NII_Ha_NII_epoch_{}'.format(epoch))
        print('SAVED EPOCH: {}'.format(epoch))
        if inference._converged(epoch, 20):
            break
