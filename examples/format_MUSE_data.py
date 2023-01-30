import numpy as np
import pandas as pd
from astropy.io import fits

import astropy.units as u
from CloudyGalaxyInference.galactic_extinction_correction import (
    galactic_extinction_correction,
)

# Inputs
redshift = 0.005224
ra = 185.4788625  # in degrees
dec = 4.4737744  # in degrees
line_wavelengths = np.array([4862.0, 4960.0, 5008.0, 6549.0, 6564.0, 6585.0])
obs_line_wavelengths = line_wavelengths * (1 + redshift)
MUSE_cube = fits.open(
    "/Users/dirk/Documents/PhD/scripts/catalogs/MUSE/ADP.2021-07-16T10_20_56.494.fits"
)  # NGC 4303

# Calculations of correction factors due to galactic foreground dust attenuation. Using the O'Donnell 94 correction scheme and SFD galactic dust maps.
foreground_correction_factors = galactic_extinction_correction(
    ra * u.degree,
    dec * u.degree,
    obs_line_wavelengths * u.angstrom,
    np.ones_like(line_wavelengths) * (u.erg * u.cm**-2 * u.s**-1),
).value

cube_dims = MUSE_cube[6].data.shape
MUSE_df = pd.DataFrame()

MUSE_df["INDEX_I"] = np.meshgrid(np.arange(cube_dims[0]), np.arange(cube_dims[1]))[
    0
].reshape(-1)
MUSE_df["INDEX_J"] = np.meshgrid(np.arange(cube_dims[0]), np.arange(cube_dims[1]))[
    1
].reshape(-1)

MUSE_df["H_BETA_FLUX"] = (
    MUSE_cube[6].data.reshape(-1) * foreground_correction_factors[0]
)
MUSE_df["H_BETA_FLUX_ERR"] = (
    MUSE_cube[7].data.reshape(-1) * foreground_correction_factors[0]
)

MUSE_df["OIII_4959_FLUX"] = (
    MUSE_cube[12].data.reshape(-1) * foreground_correction_factors[1]
)
MUSE_df["OIII_4959_FLUX_ERR"] = (
    MUSE_cube[13].data.reshape(-1) * foreground_correction_factors[1]
)

MUSE_df["OIII_5007_FLUX"] = (
    MUSE_cube[18].data.reshape(-1) * foreground_correction_factors[2]
)
MUSE_df["OIII_5007_FLUX_ERR"] = (
    MUSE_cube[19].data.reshape(-1) * foreground_correction_factors[2]
)

MUSE_df["NII_6548_FLUX"] = (
    MUSE_cube[24].data.reshape(-1) * foreground_correction_factors[3]
)
MUSE_df["NII_6548_FLUX_ERR"] = (
    MUSE_cube[25].data.reshape(-1) * foreground_correction_factors[3]
)

MUSE_df["H_ALPHA_FLUX"] = (
    MUSE_cube[30].data.reshape(-1) * foreground_correction_factors[4]
)
MUSE_df["H_ALPHA_FLUX_ERR"] = (
    MUSE_cube[31].data.reshape(-1) * foreground_correction_factors[4]
)

MUSE_df["NII_6584_FLUX"] = (
    MUSE_cube[36].data.reshape(-1) * foreground_correction_factors[5]
)
MUSE_df["NII_6584_FLUX_ERR"] = (
    MUSE_cube[37].data.reshape(-1) * foreground_correction_factors[5]
)

MUSE_df = MUSE_df.fillna(0.0)

MUSE_df.to_csv("./MUSE_df_NGC_4303.csv")
