import numpy as np
import torch

from sbi import utils as utils
from sbi.inference import SNPE, simulate_for_sbi


def train_MUSE(
    simulation,
    save_model_name,
    save_path="./",
    num_simulations=1e6,
    prior_lower_boundary=[1.0, -1.0, -4.0, 0.1, -2.0],
    prior_upper_boundary=[5.0, 0.7, -1.0, 0.6, 0.6],
):
    """
    Function to train a SBI model on MUSE input data.
    :param simulation: simulation taking input parameter theta(torch array with free parameters (amplitude, Z, U, xi, tau))
    :param save_model_name: Name to save the trained model under
    :param save_path: location to save model
    :param num_simulations: Number of simulations used to train the posterior
    :param prior_lower_boundary: list of prior lower boundaries
    :param prior_upper_boundary: list of prior upper boundaries
    :return:
    """
    prior = utils.BoxUniform(
        low=torch.tensor(prior_lower_boundary), high=torch.tensor(prior_upper_boundary)
    )

    theta, x = simulate_for_sbi(
        simulation, proposal=prior, num_simulations=num_simulations
    )
    inference = SNPE(prior=prior)
    inference = inference.append_simulations(theta, x)

    save_epochs = np.arange(5, 500, 5)

    for epoch in save_epochs:
        if epoch == save_epochs[0]:
            density_estimator = inference.train(
                max_num_epochs=epoch, resume_training=False
            )
        else:
            density_estimator = inference.train(
                max_num_epochs=epoch, resume_training=True
            )
        posterior = inference.build_posterior(density_estimator)
        torch.save(
            posterior.net, save_path + save_model_name + "epoch_{}".format(epoch)
        )
        print("SAVED EPOCH: {}".format(epoch))
        if inference._converged(epoch, 20):
            break


def train(
    simulation,
    save_model_name,
    save_path="./",
    num_simulations=1e6,
    prior_lower_boundary=[1.0, -1.0, -4.0, 0.1, -2.0],
    prior_upper_boundary=[5.0, 0.7, -1.0, 0.6, 0.6],
):
    """
    Function to train a SBI model on input data.
    :param simulation: simulation taking input parameter theta(torch array with free parameters (amplitude, Z, U, xi, tau))
    :param save_model_name: Name to save the trained model under
    :param save_path: location to save model
    :param num_simulations: Number of simulations used to train the posterior
    :param prior_lower_boundary: list of prior lower boundaries
    :param prior_upper_boundary: list of prior upper boundaries
    :return:
    """
    prior = utils.BoxUniform(
        low=torch.tensor(prior_lower_boundary), high=torch.tensor(prior_upper_boundary)
    )

    theta, x = simulate_for_sbi(
        simulation, proposal=prior, num_simulations=num_simulations
    )
    inference = SNPE(prior=prior)
    inference = inference.append_simulations(theta, x)

    save_epochs = np.arange(5, 500, 5)

    for epoch in save_epochs:
        if epoch == save_epochs[0]:
            density_estimator = inference.train(
                max_num_epochs=epoch, resume_training=False
            )
        else:
            density_estimator = inference.train(
                max_num_epochs=epoch, resume_training=True
            )
        posterior = inference.build_posterior(density_estimator)
        torch.save(
            posterior.net, save_path + save_model_name + "epoch_{}".format(epoch)
        )
        print("SAVED EPOCH: {}".format(epoch))
        if inference._converged(epoch, 20):
            break
