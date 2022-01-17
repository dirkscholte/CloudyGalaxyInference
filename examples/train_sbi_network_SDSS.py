import numpy as np
from astropy.table import Table
import pandas as pd
import torch
torch.set_num_threads(4)

from CloudyGalaxyInference.interpolate_model_grid import InterpolateModelGrid
from CloudyGalaxyInference.gaussian_noise_model import GaussianNoiseModelWavelength
from CloudyGalaxyInference.simulate import simulation_SDSS
from CloudyGalaxyInference.train import train

# SDSS data
data_path = "/Users/dirk/Documents/PhD/scripts/catalogs/"
sdss_spec_info = Table.read(data_path+"galSpecInfo-dr8.fits")
names = [name for name in sdss_spec_info.colnames if len(sdss_spec_info[name].shape) <= 1]
sdss_spec_info = sdss_spec_info[names].to_pandas()
sdss_spec_line = Table.read(data_path+"galSpecLine-dr8.fits").to_pandas()
redshift = sdss_spec_info["Z"]

# Specify lines to be used
line_labels = ['OII_3726', 'OII_3729', 'H_BETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'H_ALPHA', 'NII_6584']
line_flux_labels = [label+'_FLUX' for label in line_labels]
line_flux_err_labels = [label+'_FLUX_ERR' for label in line_labels]
line_wavelengths = np.array([3726., 3729., 4862., 4960., 5008., 6549., 6564., 6585.])
obs_line_wavelength = np.expand_dims(line_wavelengths, axis=0) * (1 + np.expand_dims(redshift, axis=-1))

# Specify location of photoionization models and output models
model_path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'
model_name = 'test_model_SDSS_10k_lintau_'
num_simulations = 10000

# Import photoionization models
model_labels = list(np.load(model_path + 'full_model_high_res_age_2Myr_unattenuated_emission_line_labels.npy'))
model_flux = np.load(model_path + 'full_model_high_res_age_2Myr_unattenuated_emission_line_luminosity_file.npy')
model_parameters = np.load(model_path + 'full_model_high_res_age_2Myr_unattenuated_parameters_file.npy')
model_derived_parameters = np.load(model_path + 'full_model_high_res_age_2Myr_unattenuated_derived_parameters_file.npy')

# Interpolate photoionization models
interpolated_grid = InterpolateModelGrid(model_labels, model_flux, model_parameters, model_derived_parameters, normalize_by='H__1_656281A')
interpolated_flux = interpolated_grid.interpolate_flux(['O__2_372603A', 'O__2_372881A', 'H__1_486133A', 'O__3_495891A', 'O__3_500684A', 'N__2_654800A', 'H__1_656281A', 'N__2_658345A'])

# Create gaussian noise model
gaussian_noise_model = GaussianNoiseModelWavelength(sdss_spec_line[line_flux_err_labels].to_numpy().reshape(-1), obs_line_wavelength.reshape(-1))

def simulation(theta):
    return simulation_SDSS(theta, line_wavelengths, interpolated_flux, 'random', gaussian_noise_model)

train(simulation, model_name, num_simulations=num_simulations)
