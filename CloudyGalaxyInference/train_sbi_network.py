import numpy as np
import pandas as pd
from astropy.table import Table
import astropy.units as u
import torch

from galactic_extinction_correction import galactic_extinction_correction, rest_to_obs_wavelength

from interpolate_model_grid import InterpolateModelGrid
from gaussian_noise_model import GaussianNoiseModel

from sbi import utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, simulate_for_sbi


# Train model for 'OII_3726', 'OII_3729', 'HBETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'HALPHA', 'NII_6584', 'SII_6716', 'SII_6731'

#import photoionization models
path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'
model_labels = list(np.load(path + 'test_model_high_res_age_2Myr_unattenuated_emission_line_labels.npy'))
model_flux = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_emission_line_luminosity_file.npy')
model_parameters = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_parameters_file.npy')
model_derived_parameters = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_derived_parameters_file.npy')

interpolated_grid = InterpolateModelGrid(model_labels, model_flux, model_parameters, model_derived_parameters, normalize_by='H__1_656281A')
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
sf_mask = ~[np.log10(denali_fastspec["OIII_5007_FLUX"]/denali_fastspec["HBETA_FLUX"]) > 0.61*(np.log10(denali_fastspec["NII_6584_FLUX"]/denali_fastspec["HALPHA_FLUX"]) + 0.05)**-1 + 1.3][0]
line_num_mask = np.sum(denali_fastspec[line_flux_labels].to_numpy()!=0.0, axis=1)>=4.

denali_fastspec_hdu2 = denali_fastspec_hdu2[gal_mask & sn_mask & z_mask & sf_mask].reset_index()
denali_fastspec = denali_fastspec[gal_mask & sn_mask & z_mask & sf_mask].reset_index()

flux_catalogue = pd.DataFrame(denali_fastspec[line_flux_labels].to_numpy(), columns=line_labels)
sn_catalogue = pd.DataFrame(denali_fastspec[line_flux_labels].to_numpy() * denali_fastspec[line_flux_ivar_labels].to_numpy()**0.5, columns=line_labels)

gaussian_noise_model = GaussianNoiseModel(flux_catalogue, sn_catalogue, line_labels, 'HALPHA')
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
    normalized_line_flux = np.zeros((len(interpolated_flux)))
    for i in range(len(interpolated_flux)):
        normalized_line_flux[i] = interpolated_flux[i](theta[1:-1])*transmission[i]

    gaussian_noise_model.set_flux_amplitude(reference_amplitude=theta[0])
    line_flux, sn_level = gaussian_noise_model.set_sn_level(normalized_line_flux)
    sn_level = sn_level
    if redshift=='random':
        redshift = np.random.uniform(low=0.0, high=0.5)
    rest_wavelength = np.array([3727.,3729.,4861.,4959.,5007.,6548.,6563.,6584.,6717.,6731.])
    obs_wavelength = (1 + redshift) * rest_wavelength
    for i in range(len(line_flux)):
        if obs_wavelength[i]>9800. or obs_wavelength[i]<3600.:
            line_flux[i] = 0.0
            sn_level[i] = 0.0
    _, line_flux_error = gaussian_noise_model.add_gaussian_noise(line_flux, sn_level)
    line_flux_and_noise, _ = gaussian_noise_model.add_gaussian_noise(line_flux, sn_level)
    tensor_out = np.expand_dims(np.hstack([line_flux_and_noise, line_flux_error]),axis=0)
    tensor_out = torch.from_numpy(tensor_out).to(torch.float32)
    return tensor_out

num_dim = 5
prior = utils.BoxUniform(low = torch.tensor([10, -1., -4., 0.1, -2.]),
                         high= torch.tensor([400, 0.7, -1., 0.6, 0.6]))


theta, x = simulate_for_sbi(simulation, proposal=prior, num_simulations=10000)
inference = SNPE(prior=prior)
inference = inference.append_simulations(theta, x)

save_epochs = np.arange(5,200,5)

for epoch in save_epochs:
    if epoch==save_epochs[0]:
        density_estimator = inference.train(max_num_epochs=epoch, resume_training=False)  # Pick `max_num_epochs` such that it does not exceed the runtime.
    else:
        density_estimator = inference.train(max_num_epochs=epoch, resume_training=True)  # Pick `max_num_epochs` such that it does not exceed the runtime.
    posterior = inference.build_posterior(density_estimator)
    torch.save(posterior.net, './sbi_inference_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_train_10000/epoch_{}'.format(epoch))
    print('SAVED EPOCH: {}'.format(epoch))
    if inference._converged(epoch, 20):
        break

'''
print(prior)
posterior = infer(simulation, prior, 'SNPE', num_simulations=1000)

torch.save(posterior.net, 'sbi_inference_model_DESI_BGS_train_1000_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_v1')
'''
