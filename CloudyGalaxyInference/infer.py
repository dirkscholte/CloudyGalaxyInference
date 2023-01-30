import numpy as np
import corner
import matplotlib.pyplot as plt
import torch

from sbi import utils as utils
from sbi.inference.base import infer

large_number = 1e10


def calc_log_dust(logtau):
    """
    Calculate the dust surface density as in Brinchmann et al. 2013
    :param logtau: Log optical depth at 5500 Angstrom
    :return: Log of dust surface density
    """
    return np.log10(0.2 * 10**logtau)


def calc_log_gas(logZ, xi, logtau):
    """
    Calculate the gas surface density as in Brinchmann et al. 2013
    :param logZ: Log metallicity in units of solar metallicity
    :param xi: Dust-to-metal ratio
    :param logtau: Log optical depth at 5500 Angstrom
    :return: Log of gas surface density
    """
    Zsun = 0.0142  # Asplund (2009) photospheric mass fraction
    return np.log10(0.2 * 10**logtau / (xi * 10**logZ * Zsun))


def prepare_input(flux, flux_error):
    """
    fill unrealistic values with 0.0
    :param flux: Numpy array of emission line fluxes
    :param flux_error: Numpy array of emission line flux errors
    :return:
    """
    for i in range(len(flux)):
        if flux_error[i] <= 0.0 or np.isinf(flux_error)[i]:
            flux_error[i] = 0.0
            flux[i] = 0.0
    output = np.expand_dims(np.concatenate([flux, flux_error]), axis=0)
    return torch.from_numpy(output)


def fit_model_to_data(
    sbi_posterior,
    data_flux,
    data_flux_error,
    interpolated_logOH,
    num_samples=10000,
    prior_lower_boundary=[0.0, -1.0, -4.0, 0.1, -2.0],
    prior_upper_boundary=[6.0, 0.7, -1.0, 0.6, 0.6],
    plotting=False,
    plot_name="test",
    save_samples=False,
):
    """
    Inference procedure to derive the 16, 50, 84 percentile intervals of the sbi_posterior parameters.
    :param sbi_posterior: Trained SBI posterior
    :param data_flux: Numpy array of emission line fluxes
    :param data_flux_error: Numpy array of emission line flux errors
    :param interpolated_logOH: The interpolated log(OH) values of the photoionization models
    :param num_samples: Number of samples to draw for inference.
    :param prior_lower_boundary: The lower boundary below which the sampled parameter values are masked
    :param prior_upper_boundary: The lower boundary below which the sampled parameter values are masked
    :param plotting: Option to output corner plot
    :param plot_name: Option to set name of corner plot
    :return:
    """
    posterior_samples = sbi_posterior.sample(
        (num_samples,), x=prepare_input(data_flux, data_flux_error)
    )
    posterior_samples = posterior_samples.numpy()

    sample_mask = (
        np.prod(
            (posterior_samples > prior_lower_boundary)[:, 1:]
            & (posterior_samples < prior_upper_boundary)[:, 1:],
            axis=1,
        )
        == 1
    )
    masked_posterior_samples = posterior_samples[sample_mask]

    parameters_out = np.ones((25)) * -999.0
    parameters_out[-1] = np.sum(sample_mask) / len(sample_mask)
    if np.sum(sample_mask) / len(sample_mask) > 0.0:
        samples_logOH = interpolated_logOH(masked_posterior_samples[:, 1:-1])
        samples_dust = calc_log_dust(masked_posterior_samples[:, 4])
        samples_gas = calc_log_gas(
            masked_posterior_samples[:, 1],
            masked_posterior_samples[:, 3],
            masked_posterior_samples[:, 4],
        )
        masked_posterior_samples = np.hstack(
            [
                masked_posterior_samples,
                np.expand_dims(samples_logOH, axis=-1),
                np.expand_dims(samples_dust, axis=-1),
                np.expand_dims(samples_gas, axis=-1),
            ]
        )
        parameters_out[:-1] = np.percentile(
            masked_posterior_samples, [16, 50, 84], axis=0
        ).T.flatten()
        if plotting:
            corner.corner(
                masked_posterior_samples,
                show_titles=True,
                quantiles=[0.16, 0.5, 0.84],
                labels=[
                    "Amplitude",
                    "Z",
                    "U",
                    "$\\xi$",
                    "$\\tau$",
                    "log(O/H)",
                    "log(dust)",
                    "log(gas)",
                ],
            )
            plt.savefig("./corner_{}.pdf".format(plot_name))
            plt.close()
        if save_samples:
            np.save("./samples_{}.npy".format(plot_name), masked_posterior_samples)
    return parameters_out, posterior_samples


def fit_model_to_dataframe(
    posterior_network,
    dataframe,
    identifier_column,
    line_flux_labels,
    line_flux_error_labels,
    output_file,
    interpolated_logOH,
    num_samples=10000,
    prior_lower_boundary=[0.0, -1.0, -4.0, 0.1, -2.0],
    prior_upper_boundary=[6.0, 0.7, -1.0, 0.6, 0.6],
    plotting=False,
    plot_name="test",
    save_samples=False,
):
    """
    Inference procedure to derive the 16, 50, 84 percentile intervals of the sbi_posterior parameters for an entire dataframe.
    :param posterior_network: SBI posterior neural network (torch)
    :param dataframe: pandas dataframe with the data.
    :param identifier_column: column with unique identifier of a row
    :param line_flux_labels: labels of columns containing line fluxes
    :param line_flux_error_labels: labels of columns containing line flux errors
    :param output_file: Name of output file
    :param interpolated_logOH: The interpolated log(OH) values of the photoionization models
    :param num_samples: Number of samples to draw for inference.
    :param prior_lower_boundary: The lower boundary below which the sampled parameter values are masked
    :param prior_upper_boundary: The lower boundary below which the sampled parameter values are masked
    :param plotting: Option to output corner plot
    :param plot_name: Option to set name of corner plot
    :return:
    """
    prior = utils.BoxUniform(
        low=torch.tensor(
            [-large_number, -large_number, -large_number, -large_number, -large_number]
        ),
        high=torch.tensor(
            [large_number, large_number, large_number, large_number, large_number]
        ),
    )

    # Create a fake simulation to instantiate a posterior
    def fake_simulation(theta):
        return np.random.normal() * torch.tensor(
            [np.ones(2 * len(line_flux_labels)).tolist()]
        )

    # Create posterior, do minimal simulations
    posterior = infer(
        fake_simulation,
        prior,
        "SNPE",
        num_simulations=10,
    )
    # Replace posterior neural net with trained neural net from file
    posterior.net = torch.load(posterior_network)
    parameters = np.ones((len(dataframe), 26)) * -999.0
    parameters[:, 0] = dataframe[identifier_column].to_numpy()
    for i in range(len(dataframe)):
        parameters[i, 1:], _ = fit_model_to_data(
            posterior,
            dataframe[line_flux_labels].to_numpy()[i],
            dataframe[line_flux_error_labels].to_numpy()[i],
            interpolated_logOH,
            num_samples=num_samples,
            prior_lower_boundary=prior_lower_boundary,
            prior_upper_boundary=prior_upper_boundary,
            plotting=plotting,
            plot_name=plot_name + "_" + str(i),
            save_samples=save_samples,
        )
        if i % 100 == 0.0:
            np.save(output_file + ".npy", parameters)

    np.save(output_file + ".npy", parameters)
    return
