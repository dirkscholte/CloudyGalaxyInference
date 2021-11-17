import numpy as np
import pandas as pd

import torch
from sbi import utils as utils
from sbi.inference.base import infer

from CloudyGalaxyInference.infer import fit_model_to_data, fit_model_to_dataframe
from CloudyGalaxyInference.interpolate_model_grid import InterpolateModelGrid

data_df = pd.read_csv('../CloudyGalaxyInference/MUSE_df_NGC_4303.csv')
data_df['SPAXELID'] = np.arange(len(data_df))
model_path = '/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/'
posterior_network = 'sbi_inference_MUSE_train_1M_Hb_OIII_OIII_NII_Ha_NII_epoch_35'
output_file = 'test_MUSE'

line_labels = ['H_BETA', 'OIII_4959', 'OIII_5007', 'NII_6548', 'H_ALPHA', 'NII_6584']
line_flux_labels = [label+'_FLUX' for label in line_labels]
line_flux_err_labels = [label+'_FLUX_ERR' for label in line_labels]
line_wavelengths = [4862., 4960., 5008., 6549., 6564., 6585.]# in Angstrom

# Import photoionization models
model_labels = list(np.load(model_path + 'test_model_high_res_age_2Myr_unattenuated_emission_line_labels.npy'))
model_flux = np.load(model_path + 'test_model_high_res_age_2Myr_unattenuated_emission_line_luminosity_file.npy')
model_parameters = np.load(model_path + 'test_model_high_res_age_2Myr_unattenuated_parameters_file.npy')
model_derived_parameters = np.load(model_path + 'test_model_high_res_age_2Myr_unattenuated_derived_parameters_file.npy')

interpolated_grid = InterpolateModelGrid(model_labels, model_flux, model_parameters, model_derived_parameters,
                                         normalize_by='H__1_656281A')
interpolated_derived_parameters = interpolated_grid.interpolate_derived_parameters()

interpolated_F, interpolated_logOH = interpolated_derived_parameters[0], interpolated_derived_parameters[1]

fit_model_to_dataframe('sbi_inference_MUSE_train_1M_Hb_OIII_OIII_NII_Ha_NII_epoch_35', data_df, 'SPAXELID', line_flux_labels, line_flux_err_labels, output_file, interpolated_logOH)
