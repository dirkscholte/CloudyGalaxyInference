import numpy as np
import torch
from scipy.special import exp1

def transmission_function(lambda_, tau, n=-1.3):
    '''
    Function to calculate the transmission function. As in Charlot and Fall (2000)
    :param lambda_: Wavelength values of the spectrum bins in Angstrom
    :param logtau: Log optical depth at 5500 Angstrom
    :param n: Exponent of power law. Default is -1.3 as is appropriate for birth clouds (-0.7 for general ISM).
    :return: Transmission function for each bin in the spectrum
    '''
    lambda_ = np.array(lambda_)
    return np.exp(-tau * (lambda_/5500)**n)

def transmission_function_slab(wl, tauV, n=-1.3):
    tau_wl = tauV * (wl/5500.)**n
    return 1/(2*tau_wl) * (1 + (tau_wl - 1) * np.exp(-tau_wl) - tau_wl**2 * exp1(tau_wl))

def simulation_MUSE(theta, line_wavelengths, interpolated_flux, redshifts, gaussian_noise_model, dust_geometry='foreground_screen'):
    '''
    Function to simulate emission line observations from photoionization models and a Gaussian noise model.
    :param theta: Input vector containing the free parameters of the model (Amplitude, Z, U, xi, tau)
    :param line_wavelengths: Numpy array with the rest wavelengths of the emission lines
    :param interpolated_flux: Interpolated emission line flux from photoionization models
    :param redshift: The redshift at which the observations are simulated.
    :param gaussian_noise_model: Gaussian noise model class
    :return:
    '''
    theta = theta.numpy()[0]
    if dust_geometry=='foreground_screen':
        transmission = transmission_function(line_wavelengths, theta[-1])
    elif dust_geometry=='slab':
        transmission = transmission_function_slab(line_wavelengths, theta[-1])
    model_line_flux = np.zeros((len(interpolated_flux)))
    for i in range(len(interpolated_flux)):
        model_line_flux[i] = 10**theta[0] * interpolated_flux[i](theta[1:-1]) * transmission[i]
    if isinstance(redshifts, str):
        if redshifts=='random':
            redshift = np.random.uniform(low=0.0, high=0.5)
        else:
            print('This redshift input is not supported!')
    else:
        redshift = np.random.choice(redshifts)
    line_flux, line_flux_error = gaussian_noise_model.add_gaussian_noise(model_line_flux, (1 + redshift) * line_wavelengths, np.ones_like(line_wavelengths) * np.random.uniform(low=0.0, high=100.0))
    tensor_out = np.expand_dims(np.hstack([line_flux, line_flux_error]),axis=0)
    tensor_out = torch.from_numpy(tensor_out).to(torch.float32)
    return tensor_out

def simulation_SDSS(theta, line_wavelengths, interpolated_flux, redshifts, gaussian_noise_model, dust_geometry='foreground_screen'):
    '''
    Function to simulate emission line observations from photoionization models and a Gaussian noise model.
    :param theta: Input vector containing the free parameters of the model (Amplitude, Z, U, xi, tau)
    :param line_wavelengths: Numpy array with the rest wavelengths of the emission lines
    :param interpolated_flux: Interpolated emission line flux from photoionization models
    :param redshift: The redshift at which the observations are simulated.
    :param gaussian_noise_model: Gaussian noise model class
    :return:
    '''
    theta = theta.numpy()[0]
    if dust_geometry=='foreground_screen':
        transmission = transmission_function(line_wavelengths, theta[-1])
    elif dust_geometry=='slab':
        transmission = transmission_function_slab(line_wavelengths, theta[-1])
    model_line_flux = np.zeros((len(interpolated_flux)))
    for i in range(len(interpolated_flux)):
        model_line_flux[i] = 10**theta[0] * interpolated_flux[i](theta[1:-1]) * transmission[i]
    if isinstance(redshifts, str):
        if redshifts=='random':
            redshift = np.random.uniform(low=0.0, high=0.5)
        else:
            print('This redshift input is not supported!')
    else:
        redshift = np.random.choice(redshifts)
    line_flux, line_flux_error = gaussian_noise_model.add_gaussian_noise(model_line_flux, (1 + redshift) * line_wavelengths, np.ones_like(line_wavelengths) * np.random.uniform(low=0.0, high=100.0))
    tensor_out = np.expand_dims(np.hstack([line_flux, line_flux_error]),axis=0)
    tensor_out = torch.from_numpy(tensor_out).to(torch.float32)
    return tensor_out
