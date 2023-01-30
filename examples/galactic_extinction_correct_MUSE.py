import pandas as pd

from CloudyGalaxyInference.galactic_extinction_correction import (
    galactic_extinction_correct_MUSE_df,
)

dataframe = pd.read_csv("")

column_rest_wls = np.array(
    [
        4862.0,
        4960.0,
        5008.0,
        6549.0,
        6564.0,
        6585.0,
        6717.0,
        6731.0,
        4862.0,
        4960.0,
        5008.0,
        6549.0,
        6564.0,
        6585.0,
        6717.0,
        6731.0,
    ]
)
column_labels = [
    "H_BETA_FLUX",
    "OIII_4959_FLUX",
    "OIII_5007_FLUX",
    "NII_6548_FLUX",
    "H_ALPHA_FLUX",
    "NII_6584_FLUX",
    "SII_6717_FLUX",
    "SII_6731_FLUX",
    "H_BETA_FLUX_ERR",
    "OIII_4959_FLUX_ERR",
    "OIII_5007_FLUX_ERR",
    "NII_6548_FLUX_ERR",
    "H_ALPHA_FLUX_ERR",
    "NII_6584_FLUX_ERR",
    "SII_6717_FLUX_ERR",
    "SII_6731_FLUX_ERR",
]

corrected_dataframe = galactic_extinction_correct_MUSE_df(
    dataframe, column_rest_wls, column_labels, galaxy_id_column="NAME"
)
