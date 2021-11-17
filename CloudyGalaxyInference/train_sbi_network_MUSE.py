import numpy as np
import pandas as pd

from interpolate_model_grid import InterpolateModelGrid
from gaussian_noise_model import GaussianNoiseModelMUSE
from simulate import simulation_MUSE
from train import train_MUSE

redshift   = 0.005224
model_path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'

line_labels = ['H_BETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'H_ALPHA', 'NII_6584']
line_flux_labels = [label+'_FLUX' for label in line_labels]
line_flux_err_labels = [label+'_FLUX_ERR' for label in line_labels]
line_wavelengths = np.array([4862., 4960., 5008., 6549., 6564., 6585.])
obs_line_wavelength = line_wavelengths * (1 + redshift)

MUSE_df = pd.read_csv('MUSE_df_NGC_4303.csv')

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

def simulation(theta):
    return simulation_MUSE(theta, line_wavelengths, interpolated_flux, redshift, gaussian_noise_model)

train_MUSE(simulation, 'test_model_MUSE', num_simulations=1e6)
