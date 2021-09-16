import numpy as np
import pandas as pd

class GaussianNoiseModel():
    def __init__(self, flux_catalogue, sn_catalogue, line_list, amplitude_reference_line):
        self.flux_catalogue = flux_catalogue # Pandas dataframe
        self.sn_catalogue = sn_catalogue # Pandas dataframe
        self.line_list = line_list
        self.amplitude_reference_line = amplitude_reference_line
        self.reference_index = np.argwhere(np.array(line_list)==amplitude_reference_line)[0]
        return

    def set_flux_amplitude(self, quantile='random', slice_width=0.2, reference_amplitude='random'):
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

    def set_sn_level(self, normalized_line_flux, slice_width=0.2):
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
            if sn_level[i]==0. or pd.isna(sn_level[i]):
                flux_error[i] = 5*self.reference_amplitude
            else:
                flux_error[i] = flux[i]/sn_level[i]
        flux_and_noise = flux + np.random.normal(loc=0.0, scale=1.0, size=len(flux)) * flux_error
        return flux_and_noise, flux_error