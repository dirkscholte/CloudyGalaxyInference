import numpy as np
from scipy.interpolate import RegularGridInterpolator

class InterpolateModelGrid():
    def __init__(self, model_line_labels, model_flux_grid, model_parameter_grid, model_derived_parameter_grid, normalize_by='Unnormalized'):
        self.model_line_labels = dict(zip(model_line_labels, range(len(model_line_labels))))
        self.model_flux_grid = model_flux_grid
        self.model_parameter_grid = model_parameter_grid
        self.model_derived_parameter_grid = model_derived_parameter_grid
        self.model_parameter_list = [np.unique(np.take(model_parameter_grid, index, axis=-1)) for index in range(model_parameter_grid.shape[-1])]
        self.normalize_by = normalize_by
        if self.normalize_by == 'Unnormalized':
            self.normalized_flux_grid = self.model_flux_grid
        else:
            normalization_index = self.model_line_labels.get(self.normalize_by)
            self.normalized_flux_grid = self.model_flux_grid/np.expand_dims(np.take(self.model_flux_grid, normalization_index, axis=-1), axis=-1)

    def interpolate_flux(self, line_labels):
        interpolated_flux = []
        for i in range(len(line_labels)):
            index = self.model_line_labels.get(line_labels[i])
            interpolated_flux.append(RegularGridInterpolator(self.model_parameter_list, np.take(self.normalized_flux_grid, index, axis=-1), bounds_error=False, fill_value=None))
        return interpolated_flux

    def interpolate_derived_parameters(self):
        interpolated_derived_parameters = []
        for i in range(self.model_derived_parameter_grid.shape[-1]):
            interpolated_derived_parameters.append(RegularGridInterpolator(self.model_parameter_list, np.take(self.model_derived_parameter_grid, i, axis=-1), bounds_error=False, fill_value=None))
        return interpolated_derived_parameters

