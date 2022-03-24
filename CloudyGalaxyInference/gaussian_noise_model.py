import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

class GaussianNoiseModel():
    def __init__(self, flux_catalogue, sn_catalogue, line_list, amplitude_reference_line):
        self.flux_catalogue = flux_catalogue # Pandas dataframe
        self.sn_catalogue = sn_catalogue # Pandas dataframe
        self.line_list = line_list
        self.amplitude_reference_line = amplitude_reference_line
        self.reference_index = np.argwhere(np.array(line_list)==amplitude_reference_line)[0]
        return

    def set_flux_amplitude(self, quantile='random', slice_width=0.1, reference_amplitude='random'):
        flux_dist = self.flux_catalogue[self.amplitude_reference_line]
        sn_dist = self.sn_catalogue[self.amplitude_reference_line]
        if quantile=='random':
            if reference_amplitude=='random':
                reference_amplitude = np.nanquantile(flux_dist, np.random.uniform(low=0.,high=1.))
            self.reference_amplitude = reference_amplitude
            flux_slice = (self.flux_catalogue[self.amplitude_reference_line] > self.reference_amplitude*(1-slice_width)) & (self.flux_catalogue[self.amplitude_reference_line] < self.reference_amplitude*(1+slice_width))
            self.sn_reference = np.nanquantile(sn_dist[flux_slice], np.random.uniform(low=0.,high=1.))
        else:
            if reference_amplitude=='random':
                reference_amplitude = np.nanquantile(flux_dist, quantile)
            self.reference_amplitude = reference_amplitude
            flux_slice = self.flux_catalogue[self.amplitude_reference_line] > self.reference_amplitude*(1-slice_width) & self.flux_catalogue[self.amplitude_reference_line] < self.reference_amplitude*(1+slice_width)
            self.sn_reference = np.nanquantile(sn_dist[flux_slice], quantile)
        return self.reference_amplitude, self.sn_reference

    def set_sn_level(self, normalized_line_flux, slice_width=0.1):
        flux = self.reference_amplitude * normalized_line_flux
        sn_level = np.zeros_like(flux)
        for i in range(len(sn_level)):
            if i == self.reference_index:
                sn_level[i] = self.sn_reference
            else:
                flux_slice = (self.flux_catalogue[self.line_list[i]] > flux[i]*(1-slice_width)) & (self.flux_catalogue[self.line_list[i]] < flux[i]*(1+slice_width))
                sn_slice = (self.sn_catalogue[self.amplitude_reference_line] > self.sn_reference * (1-slice_width)) & (self.sn_catalogue[self.amplitude_reference_line] < self.sn_reference * (1+slice_width))
                sn_level[i] = np.nanquantile(self.sn_catalogue[self.line_list[i]][flux_slice & sn_slice], np.random.uniform(low=0.,high=1.))
        return flux, sn_level

    def add_gaussian_noise(self, flux, sn_level):
        flux_error = np.zeros_like(flux)
        for i in range(len(flux)):
            if (sn_level[i] == 0. or pd.isna(sn_level[i])) and (flux[i] == 0. or pd.isna(flux[i])):
                flux_error[i] = 5 * self.reference_amplitude
                flux[i] = 0.
            elif (sn_level[i] == 0. or pd.isna(sn_level[i])) and flux[i] != 0.:
                flux_error[i] = 5 * flux[i]
            else:
                flux_error[i] = flux[i] / sn_level[i]
        flux_and_noise = flux + np.random.normal(loc=0.0, scale=1.0, size=len(flux)) * flux_error
        return flux_and_noise, flux_error


class GaussianNoiseModelMUSE():
    def __init__(self, flux_error_cat, observed_wavelength_cat):
        self.flux_error_cat = flux_error_cat
        self.observed_wavelength_cat = observed_wavelength_cat
        self.noise_model = self.create_noise_model()
        return

    def create_noise_model(self):
        wl_sample = np.linspace(4000., 10000., 1000)
        percentile_sample = np.linspace(0.0,100.0,10)
        mask = (self.flux_error_cat > 0.0) & (self.flux_error_cat < 1000.0)
        errors_out = running_percentile(wl_sample, self.observed_wavelength_cat[mask], self.flux_error_cat[mask], 1*(wl_sample[1] - wl_sample[0]), percentiles=percentile_sample, minimum_data=1, fill_value=0.0)
        return RegularGridInterpolator((wl_sample, percentile_sample), errors_out, bounds_error=False, fill_value=0.0, method='nearest')

    def add_gaussian_noise(self, input_flux, input_wavelength, noise_percentile):
        flux_error = self.noise_model(np.stack([input_wavelength, noise_percentile], axis=1))
        output_flux = input_flux + np.random.normal(loc=0.0, scale=1.0, size=len(input_flux)) * flux_error
        output_flux[flux_error == 0.0] = 0.0
        return output_flux, flux_error

    def plot_noise(self):
        wl_sample = np.arange(3600., 9800., 1.0)
        #plt.scatter(self.observed_wavelength_cat, self.flux_error_cat)
        plt.plot(wl_sample, self.noise_model(np.stack([wl_sample, np.ones_like(wl_sample)*4.6], axis=1)), linewidth=1, alpha=0.2)
        plt.plot(wl_sample, self.noise_model(np.stack([wl_sample, np.ones_like(wl_sample)*16.], axis=1)), linewidth=1, alpha=0.2)
        plt.plot(wl_sample, self.noise_model(np.stack([wl_sample, np.ones_like(wl_sample)*50.], axis=1)), linewidth=1, alpha=0.2)
        plt.plot(wl_sample, self.noise_model(np.stack([wl_sample, np.ones_like(wl_sample)*84.], axis=1)), linewidth=1, alpha=0.2)
        plt.plot(wl_sample, self.noise_model(np.stack([wl_sample, np.ones_like(wl_sample)*95.4], axis=1)), linewidth=1, alpha=0.2)
        plt.savefig('gaussian_noise_model_diagnostics.png')

def running_percentile(x_sample, x, y, width, percentiles=[16,50,84], minimum_data=10, fill_value=np.nan):
    percentiles_out = np.zeros((len(x_sample), len(percentiles))) * fill_value
    for i in range(len(x_sample)):
        mask = ((x > (x_sample[i] - width/2)) & (x < (x_sample[i] + width/2)))
        if np.sum(mask)>minimum_data:
            percentiles_out[i] = np.nanpercentile(y[mask], percentiles)
    return percentiles_out

class GaussianNoiseModelWavelength():
    def __init__(self, flux_error_cat, observed_wavelength_cat, wl_range=[3600.0, 9800.0]):
        self.flux_error_cat = flux_error_cat
        self.observed_wavelength_cat = observed_wavelength_cat
        self.wl_range = wl_range
        self.noise_model = self.create_noise_model()
        return

    def create_noise_model(self):
        wl_sample = np.linspace(self.wl_range[0], self.wl_range[1], 1000)
        percentile_sample = np.linspace(0.0,100.0,10)
        mask = (self.flux_error_cat > 0.0) & (self.flux_error_cat < 1000.0)
        errors_out = running_percentile(wl_sample, self.observed_wavelength_cat[mask], self.flux_error_cat[mask], 1*(wl_sample[1] - wl_sample[0]), percentiles=percentile_sample, minimum_data=1, fill_value=0.0)
        return RegularGridInterpolator((wl_sample, percentile_sample), errors_out, bounds_error=False, fill_value=0.0, method='nearest')

    def add_gaussian_noise(self, input_flux, input_wavelength, noise_percentile):
        flux_error = self.noise_model(np.stack([input_wavelength, noise_percentile], axis=1))
        output_flux = input_flux + np.random.normal(loc=0.0, scale=1.0, size=len(input_flux)) * flux_error
        output_flux[flux_error == 0.0] = 0.0
        return output_flux, flux_error

    def plot_noise(self):
        wl_sample = np.arange(3600., 9800., 1.0)
        #plt.scatter(self.observed_wavelength_cat, self.flux_error_cat)
        #plt.plot(wl_sample, self.noise_model(np.stack([wl_sample, np.ones_like(wl_sample)*4.6], axis=1)), linewidth=1, alpha=0.2)
        plt.plot(wl_sample, self.noise_model(np.stack([wl_sample, np.ones_like(wl_sample)*16.], axis=1)), linewidth=1, alpha=0.2)
        plt.plot(wl_sample, self.noise_model(np.stack([wl_sample, np.ones_like(wl_sample)*50.], axis=1)), linewidth=1, alpha=0.2)
        plt.plot(wl_sample, self.noise_model(np.stack([wl_sample, np.ones_like(wl_sample)*84.], axis=1)), linewidth=1, alpha=0.2)
        #plt.plot(wl_sample, self.noise_model(np.stack([wl_sample, np.ones_like(wl_sample)*95.4], axis=1)), linewidth=1, alpha=0.2)
        plt.savefig('gaussian_noise_model_diagnostics.png')

def running_percentile(x_sample, x, y, width, percentiles=[16,50,84], minimum_data=10, fill_value=np.nan):
    percentiles_out = np.zeros((len(x_sample), len(percentiles))) * fill_value
    for i in range(len(x_sample)):
        mask = ((x > (x_sample[i] - width/2)) & (x < (x_sample[i] + width/2)))
        if np.sum(mask)>minimum_data:
            percentiles_out[i] = np.nanpercentile(y[mask], percentiles)
    return percentiles_out
