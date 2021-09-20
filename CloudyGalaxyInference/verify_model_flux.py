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

# Verify model for 'OII_3726', 'OII_3729', 'HBETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'HALPHA', 'NII_6584'

#import photoionization models
path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'
model_labels = list(np.load(path + 'test_model_high_res_age_2Myr_logtau_emission_line_labels.npy'))
model_flux = np.load(path + 'test_model_high_res_age_2Myr_logtau_emission_line_luminosity_file.npy')
model_parameters = np.load(path + 'test_model_high_res_age_2Myr_logtau_parameters_file.npy')
model_derived_parameters = np.load(path + 'test_model_high_res_age_2Myr_logtau_derived_parameters_file.npy')

interpolated_grid = InterpolateModelGrid(model_labels, model_flux, model_parameters, model_derived_parameters, normalize_by='H__1_656281A')
interpolated_flux = interpolated_grid.interpolate_flux(['O__2_372603A', 'O__2_372881A', 'H__1_486133A', 'O__3_495891A', 'O__3_500684A', 'N__2_654800A', 'H__1_656281A', 'N__2_658345A', 'S__2_671644A', 'S__2_673082A'])
interpolated_derived_parameters = interpolated_grid.interpolate_derived_parameters()
interpolated_logOH, interpolated_dust, interpolated_gas = interpolated_derived_parameters[1], interpolated_derived_parameters[2], interpolated_derived_parameters[3]

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

denali_fastspec_hdu2 = denali_fastspec_hdu2[gal_mask & sn_mask & z_mask & sf_mask & line_num_mask].reset_index()[0:1000]
denali_fastspec = denali_fastspec[gal_mask & sn_mask & z_mask & sf_mask & line_num_mask].reset_index()[0:1000]

print(len(denali_fastspec))

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

print(data_df)

#Fill undetected emission lines with noise
def infer_denali(flux, flux_error):
    for i in range(len(flux)):
        if flux[i] == 0. or flux_error[i] == 0. or np.isinf(flux_error)[i]:
            flux_error[i] = 5 * np.max(flux)
            flux[i] = np.random.normal() * flux_error[i]
    output = np.expand_dims(np.concatenate([flux, flux_error]), axis=0)
    return torch.from_numpy(output)

#Define number of dimensions and prior
num_dim = 5

large_number=1e10
prior = utils.BoxUniform(low = torch.tensor([-large_number, -large_number, -large_number, -large_number, -large_number]),
                         high= torch.tensor([large_number, large_number, large_number, large_number, large_number]))


#Create a fake simulation to instantiate a posterior
def fake_simulation(theta):
    return torch.tensor([[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]])

#Create posterior, do minimal simulations
posterior = infer(fake_simulation, prior, 'SNPE', num_simulations=10, )
#Replace posterior neural net with trained neural net from file
posterior.net = torch.load('sbi_inference_model_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_v7')

data_inputs = data_df["TARGETID"]

#Model fitting function
def fit_model_to_df(index, prior_min=np.array([[-large_number, -1., -4., 0.1, -2.]]), prior_max=np.array([[large_number, 0.7, -1, 0.6, 0.6]]), plotting=False):
    parameters_out = np.ones((46)) * -999.
    galaxy = data_df[data_df['TARGETID'] == index]
    parameters_out[0] = galaxy['TARGETID'].to_numpy()[0]
    data_flux = galaxy[
        ['OII_3726_FLUX', 'OII_3729_FLUX', 'HBETA_FLUX', 'OIII_4959_FLUX', 'OIII_5007_FLUX', 'NII_6548_FLUX',
         'HALPHA_FLUX', 'NII_6584_FLUX', 'SII_6716_FLUX', 'SII_6731_FLUX']].to_numpy()[0]
    data_flux_error = galaxy[
        ['OII_3726_FLUX_ERR', 'OII_3729_FLUX_ERR', 'HBETA_FLUX_ERR', 'OIII_4959_FLUX_ERR', 'OIII_5007_FLUX_ERR',
         'NII_6548_FLUX_ERR', 'HALPHA_FLUX_ERR', 'NII_6584_FLUX_ERR', 'SII_6716_FLUX_ERR',
         'SII_6731_FLUX_ERR']].to_numpy()[0]

    posterior_samples = posterior.sample((10000,), x=infer_denali(data_flux,data_flux_error))
    posterior_samples = posterior_samples.numpy()

    sample_mask = np.prod((posterior_samples > prior_min)[:,1:] & (posterior_samples < prior_max)[:,1:], axis=1)==1
    masked_posterior_samples = posterior_samples[sample_mask]

    if np.sum(sample_mask)/len(sample_mask) > 0.5:
        masked_samples_flux = np.ones((len(masked_posterior_samples),len(interpolated_flux)))*-999.
        for i in range(len(interpolated_flux)):
            masked_samples_flux[:,i] = interpolated_flux[i](masked_posterior_samples[:,1:])
        print(masked_posterior_samples.shape, masked_samples_flux.shape)
        masked_posterior_samples = np.hstack([masked_posterior_samples, masked_samples_flux])
        parameters_out[1:] = np.percentile(masked_posterior_samples, [16, 50, 84], axis=0).T.flatten()
        if plotting:
            corner.corner(masked_posterior_samples, show_titles=True, quantiles=[0.16, 0.5, 0.84], labels=['Amplitude', 'Z', 'U', '$\\xi$', '$\\tau$', 'log(O/H)', 'log(dust)', 'log(gas)'])
            plt.savefig('./plots/corner_{}.pdf'.format(int(parameters_out[0])))
            plt.close()
    return parameters_out



param_out = np.ones((len(data_df),46)) *-999.
for i in range(len(data_df)):
    print(i)
    param_out[i] = fit_model_to_df(data_df['TARGETID'][i])

print(param_out[:,1:].shape)
print(param_out)
print(data_df.shape)
data_df[['Amp_out_p16', 'Amp_out_p50', 'Amp_out_p84',
         'logZ_out_p16', 'logZ_out_p50', 'logZ_out_p84',
         'logU_out_p16', 'logU_out_p50', 'logU_out_p84',
         'xi_out_p16', 'xi_out_p50', 'xi_out_p84',
         'logtau_out_p16', 'logtau_out_p50', 'logtau_out_p84',
         'OII_3726_out_p16', 'OII_3726_out_p50', 'OII_3726_out_p84',
         'OII_3729_out_p16', 'OII_3729_out_p50', 'OII_3729_out_p84',
         'HBETA_out_p16', 'HBETA_out_p50', 'HBETA_out_p84',
         'OIII_4959_out_p16', 'OIII_4959_out_p50', 'OIII_4959_out_p84',
         'OIII_5007_out_p16', 'OIII_5007_out_p50', 'OIII_5007_out_p84',
         'NII_6548_out_p16', 'NII_6548_out_p50', 'NII_6548_out_p84',
         'HALPHA_out_p16', 'HALPHA_out_p50', 'HALPHA_out_p84',
         'NII_6584_out_p16', 'NII_6584_out_p50', 'NII_6584_out_p84',
         'SII_6716_out_p16', 'SII_6716_out_p50', 'SII_6716_out_p84',
         'SII_6731_out_p16', 'SII_6731_out_p50', 'SII_6731_out_p84'
         ]] = param_out[:,1:]


data_df.to_csv('flux_in_vs_flux_out_sbi_inference_model_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_v7')

data_df = pd.read_csv('flux_in_vs_flux_out_sbi_inference_model_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_v7')

data_df = data_df[data_df['logZ_out_p50']!=-999.]

fig, axs = plt.subplots(2,5, figsize=(10,4))

axs = axs.ravel()

for i in range(len(axs)):
    x_vals = data_df[line_labels[i]+'_FLUX']/data_df['HALPHA_FLUX']
    y_vals = data_df[line_labels[i]+'_out_p50']/data_df['HALPHA_out_p50']
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    axs[i].scatter(data_df[line_labels[i]+'_FLUX']/data_df['HALPHA_FLUX'], data_df[line_labels[i]+'_out_p50']/data_df['HALPHA_out_p50'], s=1, c=data_df['logtau_out_p50'])
    axs[i].set_xlim(-0.1*x_max, 1.1*x_max)
    axs[i].set_ylim(-0.1*y_max, 1.1*y_max)
    axs[i].set_title(line_labels[i])

    line = np.linspace(x_min,x_max,100)
    axs[i].plot(line,line,c='k')

plt.tight_layout()
plt.savefig('flux_in_vs_flux_out_sbi_inference_model_DESI_BGS_OII_OII_Hb_OIII_OIII_NII_Ha_NII_SII_SII_v7.pdf')