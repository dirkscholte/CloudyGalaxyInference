import numpy as np
from astropy.table import Table
import pandas as pd
import torch

torch.set_num_threads(4)

from CloudyGalaxyInference.infer import fit_model_to_data, fit_model_to_dataframe
from CloudyGalaxyInference.interpolate_model_grid import InterpolateModelGrid

# SDSS data
data_path = "/Users/dirk/Documents/PhD/scripts/catalogs/"
sdss_spec_info = Table.read(data_path + "galSpecInfo-dr8.fits")
names = [
    name for name in sdss_spec_info.colnames if len(sdss_spec_info[name].shape) <= 1
]
sdss_spec_info = sdss_spec_info[names].to_pandas()
sdss_spec_info = sdss_spec_info[sdss_spec_info["SPECOBJID"] != b"                   "]
sdss_spec_line = Table.read(data_path + "galSpecLine-dr8.fits").to_pandas()
sdss_spec_line = sdss_spec_line[sdss_spec_line["SPECOBJID"] != b"                   "]

# Specify location of photoionization models and output data
model_path = (
    "/Users/dirk/Documents/PhD/scripts/CloudyGalaxy/models/test_model_high_res/"
)
posterior_network = "test_model_SDSS_10k_lintau_epoch_80"
output_file = "./test_SDSS_10k_lintau"

line_labels = [
    "OII_3726",
    "OII_3729",
    "H_BETA",
    "OIII_4959",
    "OIII_5007",
    "NII_6548",
    "H_ALPHA",
    "NII_6584",
]
line_flux_labels = [label + "_FLUX" for label in line_labels]
line_flux_err_labels = [label + "_FLUX_ERR" for label in line_labels]

# Import photoionization models
model_labels = list(
    np.load(
        model_path
        + "test_model_high_res_age_2Myr_unattenuated_emission_line_labels.npy"
    )
)
model_flux = np.load(
    model_path
    + "test_model_high_res_age_2Myr_unattenuated_emission_line_luminosity_file.npy"
)
model_parameters = np.load(
    model_path + "test_model_high_res_age_2Myr_unattenuated_parameters_file.npy"
)
model_derived_parameters = np.load(
    model_path + "test_model_high_res_age_2Myr_unattenuated_derived_parameters_file.npy"
)

interpolated_grid = InterpolateModelGrid(
    model_labels,
    model_flux,
    model_parameters,
    model_derived_parameters,
    normalize_by="H__1_656281A",
)
interpolated_derived_parameters = interpolated_grid.interpolate_derived_parameters()
interpolated_F, interpolated_logOH = (
    interpolated_derived_parameters[0],
    interpolated_derived_parameters[1],
)

# Perform model fit
fit_model_to_dataframe(
    posterior_network,
    sdss_spec_line,
    "SPECOBJID",
    line_flux_labels,
    line_flux_err_labels,
    output_file,
    interpolated_logOH,
)
