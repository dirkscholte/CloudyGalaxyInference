import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class CalculatePosterior:
    def __init__(self, model_flux_values, model_line_labels):
        self.model_flux_values = model_flux_values
        self.model_line_labels = dict(zip(model_line_labels, range(len(model_line_labels))))
        self.signal_to_noise_limit = 3.0
        self.detection_flux_values = np.ma.masked_invalid(np.ones((len(self.model_line_labels)))*np.nan)
        self.detection_flux_errors = np.ma.masked_invalid(np.ones((len(self.model_line_labels)))*np.nan)
        self.uplim_flux_values = np.ma.masked_invalid(np.ones((len(self.model_line_labels)))*np.nan)
        self.uplim_flux_errors = np.ma.masked_invalid(np.ones((len(self.model_line_labels)))*np.nan)
        self.prior = np.ones(self.model_flux_values.shape[:-1]) / np.sum(np.ones(self.model_flux_values.shape[:-1]))
        self.likelihood = np.ones(self.model_flux_values.shape[:-1]) * np.nan
        self.posterior = np.ones(self.model_flux_values.shape[:-1]) * np.nan

    def reset_likelihood(self):
        self.detection_flux_values = np.ma.masked_invalid(np.ones((len(self.model_line_labels))) * np.nan)
        self.detection_flux_errors = np.ma.masked_invalid(np.ones((len(self.model_line_labels))) * np.nan)
        self.uplim_flux_values = np.ma.masked_invalid(np.ones((len(self.model_line_labels))) * np.nan)
        self.uplim_flux_errors = np.ma.masked_invalid(np.ones((len(self.model_line_labels))) * np.nan)
        self.likelihood = np.ones(self.model_flux_values.shape[-1]) * np.nan
        self.posterior = np.ones(self.model_flux_values.shape[-1]) * np.nan

    def input_data(self, data_flux_values, data_flux_errors, data_line_labels):
        for i in range(len(data_line_labels)):
            idx = self.model_line_labels.get(data_line_labels[i])
            if data_flux_values[i]/data_flux_errors[i] >= self.signal_to_noise_limit:
                self.detection_flux_values[idx] = data_flux_values[i]
                self.detection_flux_errors[idx] = data_flux_errors[i]
            else:
                self.uplim_flux_values[idx] = self.signal_to_noise_limit * data_flux_errors[i]           #F. Masci 10/25/2011 Computing flux upper-limits for non-detections
                self.uplim_flux_errors[idx] = (self.signal_to_noise_limit + 2.054) * data_flux_errors[i]


    def normalize_model(self, line_label):
        if line_label == 'detections_weighted_mean':
            scaling_factor = np.expand_dims(np.average(np.expand_dims(self.detection_flux_values, axis=0) / self.model_flux_values, weights=1 / (self.detection_flux_errors**2/self.detection_flux_values**2), axis=-1), axis=-1)
            self.model_flux_values = self.model_flux_values * scaling_factor
        else:
            self.model_flux_values = self.model_flux_values / np.expand_dims(np.take(self.model_flux_values, self.model_line_labels.get(line_label), axis=-1), axis=-1)

    def normalize_data(self, line_label):
        self.uplim_flux_errors = self.uplim_flux_errors / np.expand_dims(np.take(self.detection_flux_values, self.model_line_labels.get(line_label), axis=-1), axis=-1)
        self.uplim_flux_values = self.uplim_flux_values / np.expand_dims(np.take(self.detection_flux_values, self.model_line_labels.get(line_label), axis=-1), axis=-1)
        self.detection_flux_errors = self.detection_flux_errors / np.expand_dims(np.take(self.detection_flux_values, self.model_line_labels.get(line_label), axis=-1), axis=-1)
        self.detection_flux_values = self.detection_flux_values / np.expand_dims(np.take(self.detection_flux_values, self.model_line_labels.get(line_label), axis=-1), axis=-1)

    def calculate_likelihood(self):
        def calc_lnlikelihood_detections(model, data, data_errors):
            return -0.5 * np.sum( ((data - model) / data_errors)**2, axis=-1)
        def calc_lnlikelihood_uplims(model, data, data_errors):
            return np.sum(np.log(0.5 * (erf( (data - model) / data_errors)) + 1), axis=-1)

        lnlikelihood_detections = calc_lnlikelihood_detections(self.model_flux_values,
                                                               self.detection_flux_values,
                                                               self.detection_flux_errors)
        lnlikelihood_uplims = calc_lnlikelihood_uplims(self.model_flux_values,
                                                       self.uplim_flux_values,
                                                       self.uplim_flux_errors)
        if np.sum(~self.uplim_flux_values.mask)>0:
            lnlikelihood = lnlikelihood_detections + lnlikelihood_uplims
        else:
            lnlikelihood = lnlikelihood_detections
        self.likelihood = np.exp(lnlikelihood)
        self.likelihood_detections = np.exp(lnlikelihood_detections)
        self.likelihood_uplims = np.exp(lnlikelihood_uplims)
        return self.likelihood

    def calculate_posterior(self):
        self.posterior = self.prior*self.likelihood
        self.posterior = self.posterior/np.sum(self.posterior)
        return self.posterior


class Marginalize:
    def __init__(self, parameter_values, parameter_labels, prob_dist):
        self.parameter_values = parameter_values
        self.n_parameters = len(parameter_values.shape) - 1
        self.parameter_labels = dict(zip(parameter_labels, range(len(parameter_labels))))
        self.prob_dist = prob_dist

    def get_parameter_index(self, parameter_label):
        if np.isscalar(parameter_label) and isinstance(parameter_label, str):
            return int(self.parameter_labels.get(parameter_label))
        elif np.isscalar(parameter_label) and ~isinstance(parameter_label, str):
            return int(parameter_label)
        else:
            if len(parameter_label)==1:
                if isinstance(parameter_label[0], str):
                    return int(self.parameter_labels.get(parameter_label[0]))
                else:
                    return int(parameter_label[0])
            else:
                index = np.ones((len(parameter_label))) * np.nan
                for i in range(len(parameter_label)):
                    if isinstance(parameter_label[i], str):
                        index[i] = self.parameter_labels.get(parameter_label[i])
                    elif ~isinstance(parameter_label[i], str):
                        index[i] = parameter_label[i]
                index = tuple(np.array(index, dtype=int))
                return index

    def parameter_bin_mids(self, parameter_label):
        parameter_index = self.get_parameter_index(parameter_label)
        return np.unique(np.take(self.parameter_values, parameter_index, axis=-1))

    def parameter_bin_edges(self, parameter_label):
        parameter_index = self.get_parameter_index(parameter_label)
        bin_mids = self.parameter_bin_mids(parameter_index)
        bin_edges = np.append(bin_mids - 0.5 * (bin_mids[1] - bin_mids[0]),
                              bin_mids[-1] + 0.5 * (bin_mids[1] - bin_mids[0]))
        return bin_edges

    def marginalize(self, parameter_labels):
        parameter_indices = self.get_parameter_index(parameter_labels)
        return np.sum(self.prob_dist, axis=parameter_indices)

    def marginalize_derived_parameter(self, derived_parameter_values, bin_edges):
        histogram, _ = np.histogram(derived_parameter_values, weights=self.prob_dist, bins=bin_edges)
        return histogram

    def histogram_cdf_inverse(self, quantile, bin_height, bin_edges):
        bin_width = bin_edges[1:] - bin_edges[:-1]
        bin_height = bin_height / np.sum(bin_height * bin_width)
        cumulative = np.cumsum(bin_height * bin_width)
        last_bin = np.sum([cumulative <= quantile])
        if last_bin == 0:
            remainder = quantile
        else:
            remainder = quantile - cumulative[last_bin - 1]
        frac_bin = remainder / (bin_height[last_bin] * (bin_width[last_bin]))
        cdf_inverse = bin_edges[0] + np.sum(bin_width[:last_bin]) + frac_bin * bin_width[last_bin]
        return cdf_inverse

    def parameter_percentile(self, parameter_label, percentile):
        parameter_index = self.get_parameter_index(parameter_label)
        bin_edges = self.parameter_bin_edges(parameter_index)
        parameters_to_marginalize = np.delete(np.arange(self.n_parameters), parameter_index)
        marginalized = self.marginalize(parameters_to_marginalize)
        return self.histogram_cdf_inverse(percentile/100., marginalized, bin_edges)

    def derived_parameter_percentile(self, derived_parameter_values, bin_edges, percentile):
        marginalized = self.marginalize_derived_parameter(derived_parameter_values, bin_edges)
        return self.histogram_cdf_inverse(percentile/100., marginalized, bin_edges)


class CornerPlot:
    def __init__(self,  parameter_values, parameter_labels, prob_dist, save=False, savename='cornerplot.pdf'):
        self.parameter_values = parameter_values
        self.n_parameters = len(parameter_values.shape) - 1
        self.parameter_labels = parameter_labels
        self.axis_labels = parameter_labels
        self.prob_dist = prob_dist
        self.save = save
        self.savename = savename

    def forceAspect(self, ax, aspect=1):
        im = ax.get_images()
        extent = im[0].get_extent()
        ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

    def plot(self):
        marg = Marginalize(self.parameter_values, self.parameter_labels, self.prob_dist)
        plt.figure(figsize=(6, 6))
        gs1 = gridspec.GridSpec(self.n_parameters, self.n_parameters)
        gs1.update(wspace=0.05, hspace=0.05)  # set the spacing between axes.

        for i in range(self.n_parameters ** 2):
            yi = int(i % self.n_parameters)
            xi = int(i / self.n_parameters)
            if xi == yi:
                ax1 = plt.subplot(gs1[i])
                plt.axis('on')
                # ax1.set_xticklabels([])
                # ax1.set_yticklabels([])
                ax1.tick_params(axis="x", direction="in")
                ax1.tick_params(axis="y", direction="in")
                ax1.set_xlim([marg.parameter_bin_edges(xi)[0], marg.parameter_bin_edges(xi)[-1]])
                if xi == 0:
                    ax1.axvline(0.0, linestyle='--', c='k')
                if xi == 3:
                    ax1.axvline(1.0, linestyle='--', c='k')

                parameters_to_marginalize = tuple(np.delete(np.arange(self.n_parameters), xi))
                ax1.step(marg.parameter_bin_edges(xi),
                         np.append(marg.marginalize(parameters_to_marginalize), marg.marginalize(parameters_to_marginalize)[-1]),
                         where='post')
            elif xi > yi:
                ax1 = plt.subplot(gs1[i])
                plt.axis('on')
                # ax1.set_xticklabels([])
                # ax1.set_yticklabels([])
                ax1.set_aspect(1)
                ax1.tick_params(axis="x", direction="in")
                ax1.tick_params(axis="y", direction="in")

                pars_to_marginalize = tuple(np.delete(np.arange(self.n_parameters), [xi, yi]))
                ax1.imshow(np.transpose(marg.marginalize(pars_to_marginalize)), origin='lower', cmap='Blues',
                           extent=[marg.parameter_bin_edges(yi)[0], marg.parameter_bin_edges(yi)[-1], marg.parameter_bin_edges(xi)[0],
                                   marg.parameter_bin_edges(xi)[-1]])
                self.forceAspect(ax1, aspect=1)

            else:
                ax1 = plt.subplot(gs1[i])
                plt.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax1.set_aspect(1)
                ax1.tick_params(axis="x", direction="in")
                ax1.tick_params(axis="y", direction="in")

            if yi == 0 and xi != 0:
                ax1 = plt.subplot(gs1[i])
                ax1.set_ylabel(self.axis_labels[xi])
            else:
                ax1.set_yticklabels([])

            if xi == self.n_parameters - 1:
                ax1 = plt.subplot(gs1[i])
                ax1.set_xlabel(self.axis_labels[yi])
            else:
                ax1.set_xticklabels([])
        if self.save:
            plt.savefig(self.savename)
        else:
            plt.show()

class LineFitPlot:
    def __init__(self, parameter_values, parameter_labels, model_flux_values, model_line_labels, data_flux_values, data_flux_errors, data_line_labels, show_line_labels, normalize_label, normalize_model_weighted_mean=False, parameter_colorbar=0, figsize=(10,6), ylims=None, save=False, savename='cornerplot.pdf'):
        self.parameter_values = parameter_values
        self.n_parameters = len(parameter_values.shape) - 1
        self.parameter_labels = parameter_labels
        self.model_flux_values = model_flux_values
        self.model_line_labels = dict(zip(model_line_labels, range(len(model_line_labels))))
        self.data_flux_values = data_flux_values
        self.data_flux_errors = data_flux_errors
        self.data_line_labels = data_line_labels
        self.show_line_labels = show_line_labels
        self.normalize_label = normalize_label
        self.normalize_model_weighted_mean = normalize_model_weighted_mean
        self.parameter_colorbar = parameter_colorbar
        self.figsize = figsize
        self.ylims = ylims
        self.save = save
        self.savename = savename

        self.calc_post = CalculatePosterior(self.model_flux_values, model_line_labels)
        self.calc_post.normalize_model(self.normalize_label)
        self.calc_post.input_data(self.data_flux_values, self.data_flux_errors, self.data_line_labels)
        self.calc_post.normalize_data(self.normalize_label)
        if self.normalize_model_weighted_mean:
            self.calc_post.normalize_model('detections_weighted_mean')
        self.calc_post.calculate_likelihood()
        self.calc_post.calculate_posterior()
        self.posterior = self.calc_post.posterior

        self.marg = Marginalize(self.parameter_values, self.parameter_labels, self.posterior)

    def plot(self):
        fig, ax = plt.subplots(figsize=self.figsize)

        for i in range(len(self.show_line_labels)):
            model_flux = np.take(self.calc_post.model_flux_values, self.model_line_labels.get(self.show_line_labels[i]), axis=-1).reshape(-1)
            colorbar_values = np.take(self.marg.parameter_values, self.marg.get_parameter_index(self.parameter_colorbar), axis=-1).reshape(-1)
            prob_sort = self.posterior.reshape(-1).argsort()
            pos = ax.scatter(np.random.uniform(low=0.1, high=0.9, size=model_flux.shape) + i,
                             model_flux.reshape(-1)[prob_sort[::1]],
                             alpha=0.4,
                             s=1000 * self.posterior.reshape(-1)[prob_sort[::1]] / np.max(self.posterior.reshape(-1)) + 0.3,
                             c=colorbar_values[prob_sort[::1]],
                             edgecolor='none',
                             vmin=np.min(colorbar_values),
                             vmax=np.max(colorbar_values),
                             cmap='turbo')
            if i == 0:
                ymin = np.min(model_flux)
                ymax = np.min(model_flux)
            if ymin > np.min(model_flux):
                ymin = np.min(model_flux)
            if ymax < np.max(model_flux):
                ymax = np.max(model_flux)

        for i in range(len(self.show_line_labels)):
            idx = self.model_line_labels.get(self.show_line_labels[i])
            plt.plot([0.1 + i, 0.9 + i],
                     [self.calc_post.detection_flux_values[idx], self.calc_post.detection_flux_values[idx]], c='k')
            plt.plot([0.1 + i, 0.1 + i], [self.calc_post.detection_flux_values[idx] - self.calc_post.detection_flux_errors[idx],
                                          self.calc_post.detection_flux_values[idx] + self.calc_post.detection_flux_errors[idx]], c='k')
            plt.plot([0.9 + i, 0.9 + i], [self.calc_post.detection_flux_values[idx] - self.calc_post.detection_flux_errors[idx],
                                          self.calc_post.detection_flux_values[idx] + self.calc_post.detection_flux_errors[idx]], c='k')

        for i in range(len(self.show_line_labels)):
            idx = self.model_line_labels.get(self.show_line_labels[i])
            plt.plot([0.1 + i, 0.9 + i],
                     [self.calc_post.uplim_flux_values[idx], self.calc_post.uplim_flux_values[idx]], c='grey')
            plt.plot([0.1 + i, 0.1 + i], [-999.,
                                          self.calc_post.uplim_flux_values[idx]], c='grey')
            plt.plot([0.9 + i, 0.9 + i], [-999.,
                                          self.calc_post.uplim_flux_values[idx]], c='grey')

        plt.xlim(0., len(self.show_line_labels))
        if self.ylims==None:
            plt.ylim(ymin, ymax)
        else:
            ymin, ymax = self.ylims
            plt.ylim(ymin, ymax)
        plt.yscale('log')

        plt.ylabel('Modelled flux and measured flux')
        plt.xticks(np.arange(len(self.show_line_labels)) + 0.5, labels=self.show_line_labels, fontsize=8, rotation='vertical')
        fig.colorbar(pos, label=self.parameter_colorbar)

        if self.save:
            plt.savefig(self.savename)
        else:
            plt.show()

