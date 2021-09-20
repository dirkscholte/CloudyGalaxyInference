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


#import photoionization models
path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'
model_labels = list(np.load(path + 'test_model_high_res_age_2Myr_logtau_emission_line_labels.npy'))
model_flux = np.load(path + 'test_model_high_res_age_2Myr_logtau_emission_line_luminosity_file.npy')
model_parameters = np.load(path + 'test_model_high_res_age_2Myr_logtau_parameters_file.npy')
model_derived_parameters = np.load(path + 'test_model_high_res_age_2Myr_logtau_derived_parameters_file.npy')

interpolated_grid = InterpolateModelGrid(model_labels, model_flux, model_parameters, model_derived_parameters, normalize_by='H__1_656281A')
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
    parameters_out = np.ones((25)) * -999.
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
        samples_logOH = interpolated_logOH(masked_posterior_samples[:,1:])
        samples_dust = interpolated_dust(masked_posterior_samples[:,1:])
        samples_gas = interpolated_gas(masked_posterior_samples[:,1:])
        masked_posterior_samples = np.hstack([masked_posterior_samples, np.expand_dims(samples_logOH, axis=-1), np.expand_dims(samples_dust, axis=-1), np.expand_dims(samples_gas, axis=-1)])
        parameters_out[1:] = np.percentile(masked_posterior_samples, [16, 50, 84], axis=0).T.flatten()
        if plotting:
            corner.corner(masked_posterior_samples, show_titles=True, quantiles=[0.16, 0.5, 0.84], labels=['Amplitude', 'Z', 'U', '$\\xi$', '$\\tau$', 'log(O/H)', 'log(dust)', 'log(gas)'])
            plt.savefig('./plots/corner_{}.pdf'.format(int(parameters_out[0])))
            plt.close()
    return parameters_out





parameters = np.ones((len(data_df),25)) *-999.
for i in range(len(data_df)):
    parameters[i] = fit_model_to_df(data_df['TARGETID'][i])
    if i%100 == 0.0:
        np.save('sbi_fitting_denali_full.npy', parameters)

np.save('sbi_fitting_denali_full.npy', parameters)


parameters = np.load('sbi_fitting_denali_full.npy')

print(np.sum(parameters[:,1] ==-999.))

plt.scatter(parameters[:,5], parameters[:,11], s=1, alpha=0.1)
plt.errorbar(parameters[:,5][::500], parameters[:,11][::500], xerr=[parameters[:,5][::500]-parameters[:,4][::500], parameters[:,6][::500]-parameters[:,5][::500]], yerr=[parameters[:,11][::500]-parameters[:,10][::500], parameters[:,12][::500]-parameters[:,11][::500]], zorder=0, linestyle='', c='grey')
plt.xlim(-1.,0.7)
plt.ylim(0.1,0.6)
plt.xlabel('Metallicity')
plt.ylabel('Dust-to-metal-ratio')
plt.show()

plt.scatter(parameters[:,5], np.log10(parameters[:,11]), c='k')
plt.errorbar(parameters[:,5], np.log10(parameters[:,11]), xerr=[parameters[:,5]-parameters[:,4], parameters[:,6]-parameters[:,5]], yerr=[np.log10(parameters[:,11])-np.log10(parameters[:,10]), np.log10(parameters[:,12])-np.log10(parameters[:,11])], zorder=0, linestyle='', c='grey')
plt.xlabel('Metallicity')
plt.ylabel('log(Dust-to-metal-ratio)')
plt.show()

plt.scatter(parameters[:,5], parameters[:,8], c='k')
plt.errorbar(parameters[:,5], parameters[:,8], xerr=[parameters[:,5]-parameters[:,4], parameters[:,6]-parameters[:,5]], yerr=[parameters[:,8]-parameters[:,7], parameters[:,9]-parameters[:,8]], zorder=0, linestyle='', c='grey')
plt.xlabel('Metallicity')
plt.ylabel('Ionization parameter')
plt.show()

plt.scatter(parameters[:,5], parameters[:,14], c='k')
plt.errorbar(parameters[:,5], parameters[:,14], xerr=[parameters[:,5]-parameters[:,4], parameters[:,6]-parameters[:,5]], yerr=[parameters[:,14]-parameters[:,13], parameters[:,15]-parameters[:,14]], zorder=0, linestyle='', c='grey')
plt.xlabel('Metallicity')
plt.ylabel('Dust attenuation')
plt.show()



def PP04_O3N2(NII, Ha, OIII, Hb):
    O3N2 = np.log10( (OIII/Hb) / (NII/Ha) )
    log_O_H = 8.73 - 0.32*O3N2
    logZ = log_O_H - 8.66
    return logZ

def PP04_N2(NII, Ha):
    N2 = np.log10(NII/Ha)
    log_O_H = 9.37 + 2.03 * N2 + 1.26 * N2**2 + 0.32 * N2**3
    #log_O_H = 8.90 + 0.57*N2
    logZ = log_O_H - 8.66
    return logZ

def tauV_BD(Ha, Hb):
    return 5500**-1.3/(4861.36**-1.3 - 6562.85**-1.3) * np.log((Ha/Hb)/2.86)


Z_pp04 = PP04_O3N2(denali_fastspec['NII_6584_FLUX'],denali_fastspec['HALPHA_FLUX'],denali_fastspec['OIII_5007_FLUX'],denali_fastspec['HBETA_FLUX'])
print(Z_pp04.shape)
print(parameters[:,5].shape)

zero_mask = parameters[:,5]!=0.0

line = np.linspace(-1,0.7,1000)
plt.scatter(parameters[:,5][zero_mask] + np.log10(1-parameters[:,11][zero_mask] ), Z_pp04[zero_mask] , s=1)
plt.errorbar(parameters[:,5][zero_mask]  + np.log10(1-parameters[:,11][zero_mask] ), Z_pp04[zero_mask] , xerr=[parameters[:,5][zero_mask] -parameters[:,4][zero_mask] , parameters[:,6][zero_mask] -parameters[:,5][zero_mask] ], zorder=0, linestyle='', c='grey')
plt.plot(line,line, c='k')

plt.xlabel('Photoionization metallicity')
plt.ylabel('Strong line metallicity PP04 O3N2')
plt.show()

print(np.array(parameters[:][0], dtype=int))

denali_grid_fit = pd.read_csv('/Users/dirk/Documents/PhD/scripts/CGDESIFitting/DESI_DENALI_CG_fit_2_times_error.csv')
print(denali_grid_fit)

denali_sbi_fit = pd.DataFrame(np.array(parameters[:,0], dtype=int), columns=['TARGETID'])

denali_sbi_fit['TARGETID'] = data_df['TARGETID']
denali_sbi_fit['sbi_Z_p16'] = parameters[:,4]
denali_sbi_fit['sbi_Z_p50'] = parameters[:,5]
denali_sbi_fit['sbi_Z_p84'] = parameters[:,6]
denali_sbi_fit['sbi_U_p16'] = parameters[:,7]
denali_sbi_fit['sbi_U_p50'] = parameters[:,8]
denali_sbi_fit['sbi_U_p84'] = parameters[:,9]
denali_sbi_fit['sbi_xi_p16'] = parameters[:,10]
denali_sbi_fit['sbi_xi_p50'] = parameters[:,11]
denali_sbi_fit['sbi_xi_p84'] = parameters[:,12]
denali_sbi_fit['sbi_tau_p16'] = parameters[:,13]
denali_sbi_fit['sbi_tau_p50'] = parameters[:,14]
denali_sbi_fit['sbi_tau_p84'] = parameters[:,15]
denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p16'] = parameters[:,16]
denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'] = parameters[:,17]
denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p84'] = parameters[:,18]
denali_sbi_fit['Z_pp04_O3N2'] = PP04_O3N2(data_df['NII_6584_FLUX'],data_df['HALPHA_FLUX'],data_df['OIII_5007_FLUX'],data_df['HBETA_FLUX'])
denali_sbi_fit['Z_pp04_N2'] = PP04_N2(data_df['NII_6584_FLUX'],data_df['HALPHA_FLUX'])
denali_sbi_fit['tau_BD'] = tauV_BD(data_df['HALPHA_FLUX'],data_df['HBETA_FLUX'])

line = np.linspace(-2,0.6,100)
plt.scatter(denali_sbi_fit['sbi_tau_p50'], np.log10(denali_sbi_fit['tau_BD']), s=1, alpha=0.5)
plt.errorbar(denali_sbi_fit['sbi_tau_p50'][::500], np.log10(denali_sbi_fit['tau_BD'])[::500], xerr=[denali_sbi_fit['sbi_tau_p50'][::500]-denali_sbi_fit['sbi_tau_p16'][::500],denali_sbi_fit['sbi_tau_p84'][::500]-denali_sbi_fit['sbi_tau_p50'][::500]], linestyle='', zorder=0, c='grey')
plt.plot(line,line, c='k')
plt.xlim(-2,0.6)
plt.ylim(-2,0.6)
plt.xlabel('log($\\tau_V$) [SBI]')
plt.ylabel('log($\\tau_V$) [Balmer Decrement]')
plt.show()

def running_median(x_sample, x, y, width, minimum_data=10):
    median = np.ones_like(x_sample)*np.nan
    p16 = np.ones_like(x_sample)*np.nan
    p84 = np.ones_like(x_sample)*np.nan
    for i in range(len(x_sample)):
        mask = ((x > (x_sample[i] - width)) & (x < (x_sample[i] + width)))
        if np.sum(mask)>minimum_data:
            p16[i] = np.nanpercentile(y[mask], 16)
            median[i] = np.nanpercentile(y[mask], 50)
            p84[i] = np.nanpercentile(y[mask], 84)
    return p16, median, p84

denali_sbi_fit = denali_sbi_fit[denali_sbi_fit['sbi_Z_p50'] != -999.]

fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
line = np.linspace(7.0,9.2,1000)
ax1.scatter(denali_sbi_fit['Z_pp04_O3N2']+8.66, 12+denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'], s=1, alpha=0.1)
#ax1.scatter(8.69+denali_sbi_fit['sbi_Z_p50'], denali_sbi_fit['Z_pp04']+8.66, s=1, alpha=0.1)
#ax1.errorbar(12+denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'], denali_sbi_fit['Z_pp04']+8.66, xerr = [denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50']-denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p16'],denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p84']-denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50']], linestyle='', color='grey', zorder=0)
ax1.plot(line,line, c='k')
ax1.plot(line,line, c='k', linestyle=':')
ax1.set_xlim(7.8,9.3)
ax1.set_ylim(7.8,9.3)
ax1.set_ylabel('Z [photoionization,SBI]')

ax2.scatter(denali_sbi_fit['Z_pp04_O3N2']+8.66, 12+denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'] - (denali_sbi_fit['Z_pp04_O3N2']+8.66), s=1, alpha=0.1)
x_sample = np.linspace(7.8,9.3,100)
y_sample_p16, y_sample_p50, y_sample_p84 =running_median(x_sample, denali_sbi_fit['Z_pp04_O3N2']+8.66, 12+denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'] - (denali_sbi_fit['Z_pp04_O3N2']+8.66), x_sample[1]-x_sample[0])
ax2.plot(x_sample, y_sample_p16,color='r', linestyle=':')
ax2.plot(x_sample, y_sample_p50,color='r')
ax2.plot(x_sample, y_sample_p84,color='r', linestyle=':')
ax2.set_xlabel('Z [PP04 O3N2]')
ax2.set_ylabel('Residuals')
ax2.set_xlim(7.8,9.3)
ax2.set_ylim(-0.5,0.5)
[p16,p50,p84] = np.percentile(12+denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'] - (denali_sbi_fit['Z_pp04_O3N2']+8.66), [16,50,84])
ax2.axhline(p16, color='k', linestyle = ':')
ax2.axhline(p50, color='k', linestyle = '-')
ax2.axhline(p84, color='k', linestyle = ':')
ax2.axhline(0.0, color='k', linestyle = '-')
ax2.text(0.03,0.15, 'mean: ${' + str(np.round(p50,decimals=2)) + '}_{-' + str(np.round(p50-p16,decimals=2)) + '}^{+' + str(np.round(p84-p50, decimals=2)) + '}$', transform=ax2.transAxes)
plt.subplots_adjust(hspace=0.0)
plt.show()

fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
line = np.linspace(7.0,9.2,1000)
ax1.scatter(denali_sbi_fit['Z_pp04_N2']+8.66, denali_sbi_fit['Z_pp04_O3N2']+8.66, s=1, alpha=0.1)
#ax1.scatter(8.69+denali_sbi_fit['sbi_Z_p50'], denali_sbi_fit['Z_pp04']+8.66, s=1, alpha=0.1)
#ax1.errorbar(12+denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'], denali_sbi_fit['Z_pp04']+8.66, xerr = [denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50']-denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p16'],denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p84']-denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50']], linestyle='', color='grey', zorder=0)
ax1.plot(line,line, c='k')
ax1.plot(line,line, c='k', linestyle=':')
ax1.set_xlim(7.8,9.3)
ax1.set_ylim(7.8,9.3)
ax1.set_ylabel('Z [PP04 O3N2]')

ax2.scatter(denali_sbi_fit['Z_pp04_N2']+8.66, denali_sbi_fit['Z_pp04_O3N2']+8.66 - (denali_sbi_fit['Z_pp04_N2']+8.66), s=1, alpha=0.1)
x_sample = np.linspace(7.8,9.3,100)
y_sample_p16, y_sample_p50, y_sample_p84 =running_median(x_sample, denali_sbi_fit['Z_pp04_N2']+8.66, denali_sbi_fit['Z_pp04_O3N2']+8.66 - (denali_sbi_fit['Z_pp04_N2']+8.66), x_sample[1]-x_sample[0])
ax2.plot(x_sample, y_sample_p16,color='r', linestyle=':')
ax2.plot(x_sample, y_sample_p50,color='r')
ax2.plot(x_sample, y_sample_p84,color='r', linestyle=':')
ax2.set_xlabel('Z [PP04 N2]')
ax2.set_ylabel('Residuals')
ax2.set_xlim(7.8,9.3)
ax2.set_ylim(-0.5,0.5)
[p16,p50,p84] = np.nanpercentile(denali_sbi_fit['Z_pp04_O3N2']+8.66 - (denali_sbi_fit['Z_pp04_N2']+8.66), [16,50,84])
ax2.axhline(p16, color='k', linestyle = ':')
ax2.axhline(p50, color='k', linestyle = '-')
ax2.axhline(p84, color='k', linestyle = ':')
ax2.axhline(0.0, color='k', linestyle = '-')
ax2.text(0.03,0.15, 'mean: ${' + str(np.round(p50,decimals=2)) + '}_{-' + str(np.round(p50-p16,decimals=2)) + '}^{+' + str(np.round(p84-p50, decimals=2)) + '}$', transform=ax2.transAxes)
plt.subplots_adjust(hspace=0.0)
plt.show()

fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
line = np.linspace(7.0,9.2,1000)
ax1.scatter(denali_sbi_fit['Z_pp04_N2']+8.66, 12+denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'], s=1, alpha=0.1)
#ax1.scatter(8.69+denali_sbi_fit['sbi_Z_p50'], denali_sbi_fit['Z_pp04']+8.66, s=1, alpha=0.1)
#ax1.errorbar(12+denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'], denali_sbi_fit['Z_pp04']+8.66, xerr = [denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50']-denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p16'],denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p84']-denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50']], linestyle='', color='grey', zorder=0)
ax1.plot(line,line, c='k')
ax1.plot(line,line, c='k', linestyle=':')
ax1.set_xlim(7.8,9.3)
ax1.set_ylim(7.8,9.3)
ax1.set_ylabel('Z [photoionization,SBI]')

ax2.scatter(denali_sbi_fit['Z_pp04_N2']+8.66, 12+denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'] - (denali_sbi_fit['Z_pp04_N2']+8.66), s=1, alpha=0.1)
x_sample = np.linspace(7.8,9.3,100)
y_sample_p16, y_sample_p50, y_sample_p84 =running_median(x_sample, denali_sbi_fit['Z_pp04_N2']+8.66, 12+denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'] - (denali_sbi_fit['Z_pp04_N2']+8.66), x_sample[1]-x_sample[0])
ax2.plot(x_sample, y_sample_p16,color='r', linestyle=':')
ax2.plot(x_sample, y_sample_p50,color='r')
ax2.plot(x_sample, y_sample_p84,color='r', linestyle=':')
ax2.set_xlabel('Z [PP04 N2]')
ax2.set_ylabel('Residuals')
ax2.set_xlim(7.8,9.3)
ax2.set_ylim(-0.5,0.5)
[p16,p50,p84] = np.nanpercentile(12+denali_sbi_fit['sbi_GAS_OXYGEN_ABUND_p50'] - (denali_sbi_fit['Z_pp04_N2']+8.66), [16,50,84])
ax2.axhline(p16, color='k', linestyle = ':')
ax2.axhline(p50, color='k', linestyle = '-')
ax2.axhline(p84, color='k', linestyle = ':')
ax2.axhline(0.0, color='k', linestyle = '-')
ax2.text(0.03,0.15, 'mean: ${' + str(np.round(p50,decimals=2)) + '}_{-' + str(np.round(p50-p16,decimals=2)) + '}^{+' + str(np.round(p84-p50, decimals=2)) + '}$', transform=ax2.transAxes)
plt.subplots_adjust(hspace=0.0)
plt.show()

print(denali_sbi_fit)

merged = denali_grid_fit.merge(denali_sbi_fit, on='TARGETID')

print(merged)

line = np.linspace(-1,0.7,1000)
plt.scatter(merged['GAS_OXYGEN_ABUND_p50'], merged['sbi_Z_p50'] + np.log10(1-merged['sbi_xi_p50']))
#plt.errorbar(parameters[0:1000,5] + np.log10(1-parameters[0:1000,11]), Z_pp04, xerr=[parameters[0:1000,5]-parameters[0:1000,4], parameters[0:1000,6]-parameters[0:1000,5]], zorder=0, linestyle='', c='grey')
plt.plot(line,line, c='k')

plt.xlabel('Photoionization metallicity [grid inference]')
plt.ylabel('Photoionization metallicity [simulation based inference]')
plt.show()

plt.scatter(merged['Z_p50'], merged['sbi_Z_p50'])
#plt.errorbar(parameters[0:1000,5] + np.log10(1-parameters[0:1000,11]), Z_pp04, xerr=[parameters[0:1000,5]-parameters[0:1000,4], parameters[0:1000,6]-parameters[0:1000,5]], zorder=0, linestyle='', c='grey')
plt.plot(line,line, c='k')

plt.xlabel('Photoionization metallicity [grid inference]')
plt.ylabel('Photoionization metallicity [simulation based inference]')
plt.show()

fig, (ax1,ax2,ax3) = plt.subplots(1,3)

ax1.scatter(merged['Z_p50'], merged['sbi_Z_p50'], s=1, alpha=0.1)
ax1.errorbar(merged['Z_p50'][::500], merged['sbi_Z_p50'][::500], xerr=[merged['Z_p50'][::500]-merged['Z_p16'][::500],merged['Z_p84'][::500]-merged['Z_p50'][::500]], yerr = [merged['sbi_Z_p50'][::500]-merged['sbi_Z_p16'][::500],merged['sbi_Z_p84'][::500]-merged['sbi_Z_p50'][::500]], linestyle='', color='grey', zorder=0)
ax1.plot(line,line, c='k')
ax1.set_xlabel('Photoionization metallicity [grid inference]')
ax1.set_ylabel('Photoionization metallicity [simulation based inference]')

ax2.scatter(merged['Z_p50'], merged['Z_pp04_O3N2'], s=1, alpha=0.1)
ax2.errorbar(merged['Z_p50'][::500], merged['Z_pp04_O3N2'][::500], xerr=[merged['Z_p50'][::500]-merged['Z_p16'][::500],merged['Z_p84'][::500]-merged['Z_p50'][::500]], linestyle='', color='grey', zorder=0)
ax2.plot(line,line, c='k')
ax2.set_xlabel('Photoionization metallicity [grid inference]')
ax2.set_ylabel('Strong line metallicity [PP04 O3N2]')

ax3.scatter(merged['sbi_Z_p50'], merged['Z_pp04_O3N2'], s=1, alpha=0.1)
ax3.errorbar(merged['sbi_Z_p50'][::500], merged['Z_pp04_O3N2'][::500], xerr = [merged['sbi_Z_p50'][::500]-merged['sbi_Z_p16'][::500],merged['sbi_Z_p84'][::500]-merged['sbi_Z_p50'][::500]], linestyle='', color='grey', zorder=0)
ax3.plot(line,line, c='k')
ax3.set_xlabel('Photoionization metallicity [simulation based inference]')
ax3.set_ylabel('Strong line metallicity [PP04 O3N2]')

plt.show()

