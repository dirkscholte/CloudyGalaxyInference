import numpy as np
import pandas as pd
from astropy.table import Table
import torch
import corner

import matplotlib.pyplot as plt

from interpolate_model_grid import InterpolateModelGrid
from gaussian_noise_model import GaussianNoiseModel

from sbi import utils as utils
from sbi.inference.base import infer

#import photoionization models
path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'
model_labels = list(np.load(path + 'test_model_high_res_age_2Myr_logtau_emission_line_labels.npy'))
model_flux = np.load(path + 'test_model_high_res_age_2Myr_logtau_emission_line_luminosity_file.npy')
model_parameters = np.load(path + 'test_model_high_res_age_2Myr_logtau_parameters_file.npy')
model_derived_parameters = np.load(path + 'test_model_high_res_age_2Myr_logtau_derived_parameters_file.npy')

interpolated_grid = InterpolateModelGrid(model_labels, model_flux, model_parameters, normalize_by='H__1_656281A')
interpolated_flux = interpolated_grid.interpolate_flux(['O__2_372603A', 'O__2_372881A', 'H__1_486133A', 'O__3_495891A', 'O__3_500684A', 'N__2_654800A', 'H__1_656281A', 'N__2_658345A', 'S__2_671644A', 'S__2_673082A'])

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

denali_fastspec_hdu2 = denali_fastspec_hdu2[gal_mask & sn_mask & z_mask & sf_mask].reset_index()
denali_fastspec = denali_fastspec[gal_mask & sn_mask & z_mask & sf_mask].reset_index()

flux_catalogue = pd.DataFrame(denali_fastspec[line_flux_labels].to_numpy(), columns=line_labels)
sn_catalogue = pd.DataFrame(denali_fastspec[line_flux_labels].to_numpy() * denali_fastspec[line_flux_ivar_labels].to_numpy()**0.5, columns=line_labels)

gaussian_noise_model = GaussianNoiseModel(flux_catalogue, sn_catalogue, line_labels, 'HALPHA')

def simulation(theta, redshift='random'):
    normalized_line_flux = np.zeros((len(interpolated_flux)))
    for i in range(len(interpolated_flux)):
        normalized_line_flux[i] = interpolated_flux[i](theta[1:])

    gaussian_noise_model.set_flux_amplitude(reference_amplitude=theta.numpy()[0])
    line_flux, sn_level = gaussian_noise_model.set_sn_level(normalized_line_flux)
    if redshift=='random':
        redshift = np.random.uniform(low=0.0, high=0.5)
    rest_wavelength = np.array([3727.,3729.,4861.,4959.,5007.,6548.,6563.,6584.,6717.,6731.])
    obs_wavelength = (1 + redshift) * rest_wavelength
    for i in range(len(line_flux)):
        if obs_wavelength[i]>9800. or obs_wavelength[i]<3600.:
            line_flux[i] = 0.0
            sn_level[i] = 0.0
    line_flux_and_noise, line_flux_error = gaussian_noise_model.add_gaussian_noise(line_flux, sn_level)
    return torch.cat([torch.from_numpy(line_flux_and_noise), torch.from_numpy(line_flux_error)], 0)

num_dim = 5
prior = utils.BoxUniform(low = torch.tensor([10, -1., -4., 0.1, -2.]),
                         high= torch.tensor([400, 0.7, 0.5, 0.6, 0.6]))

print(prior)
posterior = infer(simulation, prior, 'SNPE', num_simulations=100000)

torch.save(posterior.net, 'sbi_inference_model_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII')

posterior = infer(simulation, prior, 'SNPE', num_simulations=10)
posterior.net = torch.load('../models/sbi_inference_model_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII')
x_o_1s = []
for i in range(1):
    torch.manual_seed(i+11)
    theta = torch.tensor([50, 0.0,-1.43,0.3,0.0])
    x_o_1 = simulation(theta, redshift=0.48)
    x_o_1s.append(x_o_1)

print('start')
for i in range(10):
    x_o_1 = denali_fastspec[line_flux_labels+line_flux_ivar_labels].to_numpy()[30+i]

    print(x_o_1)
    posterior_samples_1 = posterior.sample((10000,), x=x_o_1)
    posterior_samples_1 = posterior_samples_1.numpy()
    print(posterior_samples_1.shape)
    gas = posterior_samples_1[:,4] - np.log10(posterior_samples_1[:,3]) - posterior_samples_1[:,1] - np.log10(0.0142)
    print(gas.shape)
    posterior_samples_1 = np.hstack([posterior_samples_1,np.expand_dims(gas, axis=-1)])

    corner.corner(posterior_samples_1, show_titles=True, quantiles=[0.16, 0.5, 0.84], labels=['Amplitude', 'Z', 'U', 'xi', 'tau', 'gas'])
    plt.show()