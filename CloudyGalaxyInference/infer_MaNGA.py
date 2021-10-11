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

line_labels = ['oii_3727', 'oii_3729', 'hb_4862', 'oiii_4960', 'oiii_5008', 'nii_6549', 'ha_6564', 'nii_6585', 'sii_6718', 'sii_6732']
line_flux_labels = [label+'_flux' for label in line_labels]
line_flux_err_labels = [label+'_flux_err' for label in line_labels]
line_wavelengths = [3727., 3729., 4862., 4960., 5008., 6549., 6564., 6585., 6718., 6732.]

#Fill analysis dataframe
data_df = pd.read_csv('../../CGMangaFitting/MaNGA_8083-12702_map_uncorrected.csv')

print(data_df.columns.to_list())


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

#Fill undetected emission lines with noise
def infer_denali(flux, flux_error):
    for i in range(len(flux)):
        if flux[i] == 0. or flux_error[i] == 0. or np.isinf(flux_error)[i]:
            flux_error[i] = 5 * np.max(flux)
            flux[i] = np.random.normal() * flux_error[i]
    output = np.expand_dims(np.concatenate([flux, flux_error]), axis=0)
    return torch.from_numpy(output)

#Model fitting function
def fit_model_to_df(index, prior_min=np.array([[-large_number, -1., -4., 0.1, -2.]]), prior_max=np.array([[large_number, 0.7, -1.0, 0.6, 0.6]]), plotting=False):
    parameters_out = np.ones((25)) * -999.
    galaxy = data_df[data_df['1D_INDEX'] == index]
    parameters_out[0] = galaxy['1D_INDEX'].to_numpy()[0]
    data_flux = galaxy[line_flux_labels].to_numpy()[0]
    data_flux_error = galaxy[line_flux_err_labels].to_numpy()[0]

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
final_epoch = [100]

for i in [0]:
    posterior.net = torch.load('./sbi_inference_larger_grid_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_train_285120/log_U_max_-1_epoch_85')

    data_inputs = data_df["1D_INDEX"]



    parameters = np.ones((len(data_df),25)) *-999.
    for j in range(len(data_df)):
        parameters[j] = fit_model_to_df(data_df['1D_INDEX'][j])
        parameters[j,0] = data_df['1D_INDEX'][j]
        if j%100 == 0.0:
            np.save(
                'MaNGA_data/sbi_fits_MaNGA_8083-12702_uncorrected.npy', parameters)

    np.save(
        'MaNGA_data/sbi_fits_MaNGA_8083-12702_uncorrected.npy', parameters)

