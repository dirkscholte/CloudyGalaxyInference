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

from scipy import stats

large_number=1e10


# Train model for 'OII_3726', 'OII_3729', 'HBETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'HALPHA', 'NII_6584', 'SII_6716', 'SII_6731'

#import photoionization models
path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'
model_labels = list(np.load(path + 'test_model_high_res_age_2Myr_unattenuated_emission_line_labels.npy'))
model_flux = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_emission_line_luminosity_file.npy')
model_parameters = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_parameters_file.npy')
model_derived_parameters = np.load(path + 'test_model_high_res_age_2Myr_unattenuated_derived_parameters_file.npy')

interpolated_grid = InterpolateModelGrid(model_labels, model_flux, model_parameters, model_derived_parameters, normalize_by='H__1_656281A')
interpolated_flux = interpolated_grid.interpolate_flux(['O__2_372603A', 'O__2_372881A', 'H__1_486133A', 'O__3_495891A', 'O__3_500684A', 'N__2_654800A', 'H__1_656281A', 'N__2_658345A', 'S__2_671644A', 'S__2_673082A'])
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
sf_mask = ~[np.log10(denali_fastspec["OIII_5007_FLUX"]/denali_fastspec["HBETA_FLUX"]) > 0.61*(np.log10(denali_fastspec["NII_6584_FLUX"]/denali_fastspec["HALPHA_FLUX"]) + 0.05)**-1 + 1.3][0]
line_num_mask = np.sum(denali_fastspec[line_flux_labels].to_numpy()!=0.0, axis=1)>=4.

denali_fastspec_hdu2 = denali_fastspec_hdu2[gal_mask & sn_mask & z_mask & sf_mask].reset_index()
denali_fastspec = denali_fastspec[gal_mask & sn_mask & z_mask & sf_mask].reset_index()

flux_catalogue = pd.DataFrame(denali_fastspec[line_flux_labels].to_numpy(), columns=line_labels)
sn_catalogue = pd.DataFrame(denali_fastspec[line_flux_labels].to_numpy() * denali_fastspec[line_flux_ivar_labels].to_numpy()**0.5, columns=line_labels)

denali_fastspec_hdu2 = denali_fastspec_hdu2[gal_mask & sn_mask & z_mask & sf_mask & line_num_mask].reset_index()[0:500]
denali_fastspec = denali_fastspec[gal_mask & sn_mask & z_mask & sf_mask & line_num_mask].reset_index()[0:500]

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

#Fill undetected emission lines with noise
def infer_denali(flux, flux_error):
    for i in range(len(flux)):
        if flux[i] == 0. or flux_error[i] == 0. or np.isinf(flux_error)[i]:
            flux_error[i] = 5 * np.max(flux)
            flux[i] = np.random.normal() * flux_error[i]
    output = np.expand_dims(np.concatenate([flux, flux_error]), axis=0)
    return torch.from_numpy(output)

#Model fitting function
def fit_model_to_df(index, prior_min=np.array([[-large_number, -1., -4., 0.1, -2.]]), prior_max=np.array([[large_number, 0.7, -1, 0.6, 0.6]]), plotting=False):
    parameters_out = np.ones((19)) * -999.
    galaxy = data_df[data_df['TARGETID'] == index]
    parameters_out[0] = galaxy['TARGETID'].to_numpy()[0]
    data_flux = galaxy[
        ['OII_3726_FLUX', 'OII_3729_FLUX', 'HBETA_FLUX', 'OIII_4959_FLUX', 'OIII_5007_FLUX', 'NII_6548_FLUX',
         'HALPHA_FLUX', 'NII_6584_FLUX', 'SII_6716_FLUX', 'SII_6731_FLUX']].to_numpy()[0]
    data_flux_error = galaxy[
        ['OII_3726_FLUX_ERR', 'OII_3729_FLUX_ERR', 'HBETA_FLUX_ERR', 'OIII_4959_FLUX_ERR', 'OIII_5007_FLUX_ERR',
         'NII_6548_FLUX_ERR', 'HALPHA_FLUX_ERR', 'NII_6584_FLUX_ERR', 'SII_6716_FLUX_ERR',
         'SII_6731_FLUX_ERR']].to_numpy()[0]

    posterior_samples = posterior_infer.sample((10000,), x=infer_denali(data_flux,data_flux_error))
    posterior_samples = posterior_samples.numpy()

    sample_mask = np.prod((posterior_samples > prior_min)[:,1:] & (posterior_samples < prior_max)[:,1:], axis=1)==1
    masked_posterior_samples = posterior_samples[sample_mask]

    if np.sum(sample_mask)/len(sample_mask) > 0.5:
        samples_logOH = interpolated_logOH(masked_posterior_samples[:,1:-1])
        masked_posterior_samples = np.hstack([masked_posterior_samples, np.expand_dims(samples_logOH, axis=-1)])
        parameters_out[1:] = np.percentile(masked_posterior_samples, [16, 50, 84], axis=0).T.flatten()
        if plotting:
            corner.corner(masked_posterior_samples, show_titles=True, quantiles=[0.16, 0.5, 0.84], labels=['Amplitude', 'Z', 'U', '$\\xi$', '$\\tau$', 'log(O/H)', 'log(dust)', 'log(gas)'])
            plt.savefig('./plots/corner_{}.pdf'.format(int(parameters_out[0])))
            plt.close()
    return parameters_out

num_dim = 5
prior = utils.BoxUniform(low = torch.tensor([10, -1., -4., 0.1, -2.]),
                         high= torch.tensor([400, 0.7, -1., 0.6, 0.6]))

prior_infer = utils.BoxUniform(low = torch.tensor([-large_number, -large_number, -large_number, -large_number, -large_number]),
                               high= torch.tensor([large_number, large_number, large_number, large_number, large_number]))
#Create a fake simulation to instantiate a posterior
def fake_simulation(theta):
    return torch.tensor([[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]])

#Create posterior, do minimal simulations
posterior_infer = infer(fake_simulation, prior_infer, 'SNPE', num_simulations=10, )

theta, x = simulate_for_sbi(simulation, proposal=prior, num_simulations=100000)
inference = SNPE(prior=prior)
inference = inference.append_simulations(theta, x)

save_epochs = np.arange(5,500,5)

epoch = 20
spearman_r = 0.0
converged = False
attempt_queue = []
spearman_queue = []
while not converged:
    if epoch==20:
        density_estimator_train = inference.train(max_num_epochs=epoch, resume_training=False)  # Pick `max_num_epochs` such that it does not exceed the runtime.
        density_estimator = density_estimator_train
    else:
        density_estimator = density_estimator_train
        density_estimator_train = inference.train(max_num_epochs=epoch, resume_training=True)  # Pick `max_num_epochs` such that it does not exceed the runtime.
    posterior = inference.build_posterior(density_estimator_train)
    posterior_infer.net = posterior.net
    parameters = np.ones((len(data_df),19)) *-999.
    for i in range(len(data_df)):
        parameters[i] = fit_model_to_df(data_df['TARGETID'][i])
    mask = parameters[:,5]!=-999.
    spearman_r_new = stats.spearmanr(parameters[:, 5][mask], parameters[:, 14][mask]).correlation
    if spearman_r_new > spearman_r:
        torch.save(posterior.net, './sbi_inference_complicated_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_train_100000/epoch_{}'.format(epoch))
        spearman_r = spearman_r_new
        print('SAVED EPOCH: {}'.format(epoch))
        print('SPEARMAN R: {}'.format(spearman_r_new))
        converged = inference._converged(epoch, 20)
        epoch += 5
    else:
        attempt_queue.append(density_estimator_train)
        spearman_queue.append(spearman_r_new)
        if len(spearman_queue)>5:
            density_estimator_train = attempt_queue[np.argmax(np.array(spearman_queue))]
            posterior = inference.build_posterior(density_estimator_train)
            torch.save(posterior.net, './sbi_inference_complicated_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_train_100000/epoch_{}'.format(epoch))
            spearman_r = np.max(np.array(spearman_queue))
            epoch += 5
        else:
            density_estimator_train = density_estimator
        converged = inference._converged(epoch, 20)
        if converged:
            torch.save(posterior.net, './sbi_inference_complicated_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_train_100000/epoch_{}'.format(epoch))

        print('SPEARMAN R DID NOT INCREASE, RETRY EPOCH: {}'.format(epoch))
        print('SPEARMAN R: {}'.format(spearman_r_new))



