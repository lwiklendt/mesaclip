import datetime
from functools import reduce
import os
import pickle

import matplotlib.pyplot as plt

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.stats

import colorednoise

import wavelet
import utils


def ratios_to_pos_and_size(ratios):
    s = np.array(ratios) / np.sum(ratios)
    return np.cumsum(s)[:-1:2], s[1::2]


def get_results(ntrains=1000):
    # cache file is just over 1GB for 1000 ntrains
    cache_filename = f'../output/snr_vs_peak_detect_cache_{ntrains}.pkl'
    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            results = pickle.load(f)
    else:

        progress_start = datetime.datetime.now()
        progress_tick = progress_start

        np.random.seed(2002121548)

        alphas = np.array([0.5, 1, 2, 4, 8, 16])
        threshes = np.array([0.3, 0.5, 0.7])
        min_cycles = np.array([2, 4, 8])
        snrs = np.r_[0.0, np.geomspace(0.1, 10, 31)]  # include SNR=0 for pure noise effect
        freqs = np.geomspace(1, 64, 101)
        t = np.linspace(0, 2, 2000)
        base_frequency = 8

        dt = t[1] - t[0]

        mother = wavelet.Morse(beta=1.58174, gam=3)
        scales = np.geomspace(2 * dt, 0.5 * (t[-1] - t[0]), 200)
        wavelet_kwargs = dict(dt=dt, scales=scales, mother=mother, syncsqz_freqs=freqs, apply_coi=True)
        freq_edges = utils.make_edges(freqs, log=True)
        log2_freq_edges = np.log2(freq_edges)
        t_edges = utils.make_edges(t)

        # to store examples if input data
        example_trains = np.zeros((len(alphas), 100, len(t)))
        example_signals = np.zeros((len(alphas), len(snrs), len(t)))  # closest one to base freq in first second
        train_freq_hists = np.zeros((len(alphas), len(freqs)))

        # to store examples of wavelet and peak_hist freqeuncy distribution
        waves = np.zeros((len(alphas), len(snrs), ntrains, len(min_cycles), len(freqs)))
        peaks = np.zeros((len(alphas), len(snrs), ntrains, len(threshes), len(freqs)))

        total_trains = len(alphas) * len(snrs) * ntrains
        elapsed_trains = 0

        for alpha_idx, alpha in enumerate(alphas):

            # isi_distribution = scipy.stats.gamma(a=regularity, scale=1/(base_frequency * regularity))

            for snr_idx, snr in enumerate(snrs):

                # used for overwritting an example signal when the spike count is closer to the expected 1/base_freq
                example_signal_dist = np.inf

                # generate spike trains
                num_resamples = 0
                for i in range(ntrains):

                    # -------------------------------------------------------------------------------------------------
                    #   generate spike train

                    needs_resample = True
                    num_resamples -= 1  # subtact the first sample because it is not a resampling but initial sample
                    train = np.zeros_like(t, dtype=bool)
                    while needs_resample:
                        num_resamples += 1

                        num_gen = int(2 * (t[-1] - t[0]) * base_frequency)  # generate more spikes than expected needed
                        inv_isi = 2**(np.random.randn(num_gen)/alpha + np.log2(base_frequency))
                        spikes = np.cumsum(1/inv_isi) - 0.5 / 8

                        # check if we have made a mistake somewhere sampling the spike train
                        if num_resamples > 0 and num_resamples % 10000 == 0:
                            print(f'    spike train {i+1} resample {num_resamples}, alpha={alpha}')
                            print(f'      spikes: {spikes}')

                        # if we got unlucky and did not generate enough to go past the end, then just try again
                        if spikes[-1] < t[-1]:
                            continue

                        # clip to recording length
                        spikes = spikes[(spikes >= t[0]) & (spikes <= t[-1])]

                        if len(spikes) > 0:
                            f = 1 / np.diff(spikes)
                            raster = np.histogram(spikes, bins=t_edges)[0]
                            if len(f) > 0 and np.all(raster < 2) and np.all(f >= freq_edges[0]) and np.all(f <= freq_edges[-1]):
                                needs_resample = False
                                train = raster > 0
                                train_freq_hists[alpha_idx, :] += np.histogram(f, bins=freq_edges)[0]

                    # store train for an example (doesn't depend on SNR, so just store from the first SNR only)
                    if snr_idx == 0 and i < example_trains.shape[1]:
                        example_trains[alpha_idx, i, :] = train

                    # -------------------------------------------------------------------------------------------------
                    #   generate noise and signal

                    signal = train.astype(float)

                    # membrane potential bumps
                    mempot = 10 * scipy.ndimage.gaussian_filter1d(signal, 0.01 / dt)

                    # make spike heights noisy
                    signal *= 1 / (1 + np.exp(colorednoise.powerlaw_psd_gaussian(0, len(t))))

                    # add mempot bumps
                    signal += mempot

                    # generate additive noise
                    noise = colorednoise.powerlaw_psd_gaussian(1, len(signal))  # use 1/f (pink) noise

                    # scale to attain desired snr
                    if snr_idx == 0:
                        # rather than making noise infinite, makes more sense to set signal to 0
                        signal *= 0
                    else:
                        # noise is scaled rather than scaling signal so that plotting signal examples looks better
                        noise *= np.sqrt(np.mean(signal**2) / snr)  # scale noise to attain desired SNR
                    signal += noise

                    # check if we should keep this signal as the example
                    num_spikes_train = np.sum(train[:len(train)//2])  # only the first 0..1s that will be plotted
                    spike_train_dist = np.abs(num_spikes_train - base_frequency)
                    if spike_train_dist < example_signal_dist:
                        # store signal as an example
                        example_signals[alpha_idx, snr_idx, :] = signal
                        example_signal_dist = spike_train_dist

                    # -------------------------------------------------------------------------------------------------
                    #   compute wavelet on both noise and signal

                    for j, k in enumerate(min_cycles):
                        w = wavelet.cwt(signal, min_cycles=k, **wavelet_kwargs)[0]
                        waves[alpha_idx, snr_idx, i, j, :] = np.mean(np.abs(w) ** 2, axis=1)

                    # -------------------------------------------------------------------------------------------------
                    #   compute peak detection on both noise and signal

                    for j in range(len(threshes)):

                        # compute on signal
                        x = signal
                        x = x - np.mean(x)
                        x = x / np.max(x)
                        spikes = utils.spike_detect(x, t, threshes[j])
                        if len(spikes) > 1:
                            f = np.log2(1 / np.diff(spikes))
                            h = np.histogram(f, bins=log2_freq_edges)[0].astype(float)
                        else:
                            h = np.zeros_like(freqs)
                        peaks[alpha_idx, snr_idx, i, j, :] = h

                    # show progress
                    now = datetime.datetime.now()
                    elapsed_trains += 1
                    if (now - progress_tick).total_seconds() > 10:
                        progress_tick = now
                        proportion_done = elapsed_trains / total_trains
                        alpha_str = f'alpha {alpha} ({alpha_idx+1}/{len(alphas)})'
                        snr_str = f'snr ({snr_idx + 1}/{len(snrs)})'
                        train_str = f'train ({i+1}/{ntrains})'
                        print(f'{alpha_str}, {snr_str}, {train_str}, {progress_str(progress_start, proportion_done)}')

        results = (alphas, threshes, min_cycles, snrs, freqs, t, base_frequency,
                   train_freq_hists, example_trains, example_signals, waves, peaks)

        with open(cache_filename, 'wb') as f:
            pickle.dump(results, f, protocol=2)

        print(f'elapsed time: {datetime.datetime.now() - progress_start}')

    return results


def zero_mu(x):
    """
    Sets the microseconds to 0.
    :param x: a timedelta or datetime object
    :return: the same type as x, with microseconds set to 0
    """
    if type(x) is datetime.datetime:
        x = x.replace(microsecond=0)
    elif type(x) is datetime.timedelta:
        # https://stackoverflow.com/a/18470628/142712
        x = x - datetime.timedelta(microseconds=x.microseconds)
    return x


def progress_str(wall_start, proportion_done):
    wall_elapsed = datetime.datetime.now() - wall_start
    wall_total_est = datetime.timedelta(seconds=wall_elapsed.total_seconds() / proportion_done)
    eta = zero_mu(wall_start + wall_total_est)
    wall_remaining = zero_mu(wall_total_est - wall_elapsed)
    return f'[{100 * proportion_done:4.1f}%], elapsed: {zero_mu(wall_elapsed)}, eta: {eta}, in: {wall_remaining}'


# pretty much copied (with minor changes) from scikits.bootstrap which wouldn't install on my system
# see https://github.com/cgevans/scikits-bootstrap/blob/master/scikits/bootstrap/bootstrap.py#L20
def bootci_pi(data, statfunc=lambda x: np.nanmean(x, axis=0), alpha=0.05, n_samples=10000):
    data = np.array(data)
    bootindexes = (np.random.randint(0, data.shape[0], size=(data.shape[0],)) for _ in range(n_samples))
    stat = np.array([statfunc(data[i]) for i in bootindexes])
    stat.sort(axis=0)
    alphas = np.array([0.5 * alpha, 1 - 0.5 * alpha])
    return stat[np.round((n_samples-1)*alphas).astype('int')]


def main_errors(ntrains):
    alphas, threshes, ks, snrs, freqs, t, base_frequency, train_freq_hists, _, _, waves, peaks = get_results(ntrains)

    # remove the alpha=0.5, doesn't really add anything
    alphas = alphas[1:]
    waves = waves[1:, ...]
    peaks = peaks[1:, ...]

    nalpha, nsnr, ntrains, nthresh, nfreq = peaks.shape
    nks = len(ks)
    log2_freqs = np.log2(freqs)

    np.random.seed(2002171330)
    nboots = 1000 if ntrains >= 1000 else 10  # only do the slow 1k in the production version of many ntrains
    bootci_kwargs = dict(statfunc=lambda _x: np.nanmean(_x, axis=0), alpha=0.05, n_samples=nboots)

    # create frequency estimate functions

    def fest_max(_w):
        return log2_freqs[np.argmax(_w)]

    def fest_median(w):
        wcs = np.cumsum(w)
        idx = np.searchsorted(wcs, wcs[-1]/2)
        return log2_freqs[idx]

    def fest_mean(w):
        return np.average(log2_freqs, weights=w)

    def fest_variance(w):
        return np.average((log2_freqs - fest_mean(w))**2, weights=w)

    fest_funcs = [fest_max, fest_median, fest_mean, fest_variance]
    fest_labels = ['ArgMax', 'Median', 'Mean', 'Variance']
    nfest = len(fest_funcs)

    wave_kwargs = [dict(facecolor=c, edgecolor=c, alpha=0.5, zorder=100 - ci)
                   for ci, c in enumerate(['0', '0.5', '0.8'])]
    peak_kwargs = [dict(facecolor=c, edgecolor=c, alpha=0.5, zorder=50 - ci)
                   for ci, c in enumerate(plt.rcParams['axes.prop_cycle'].by_key()['color'][:nthresh])]

    # axes widths and x-positions (indexed from left)
    left_margin = 0.8
    right_margin = 0.1
    width_ratios = [left_margin] + reduce((lambda a, b: a+[0.07]+b), [[1]]*len(alphas)) + [right_margin]
    xs, ws = ratios_to_pos_and_size(width_ratios)

    # axes heights and y-positions (indexed from bottom)
    top_margin = 1.1
    bottom_margin = 0.5
    height_ratios = reduce((lambda a, b: a+[0.1]+b), [[1.0]] * 2)
    height_ratios = [bottom_margin] + reduce((lambda a, b: a+[0.3]+b), [height_ratios]*nfest) + [top_margin]
    ys, hs = ratios_to_pos_and_size(height_ratios)

    fig = plt.figure(figsize=(9, 12))
    fig.text(0.5, 0.98, 'Exploration of noise and irregularity', fontsize=16, ha='center', va='center')

    for fi, (fest_func, fest_label) in enumerate(zip(fest_funcs, fest_labels)):

        # for sharing y-axes
        ax_raw = None
        ax_nrm = None

        # work-around matplotlib's inability to show ticklabels on the left-most axes in shared group
        axs = []

        for ai, alpha in enumerate(alphas):

            alpha_value_label = f'{alpha}'.rstrip('0').rstrip('.')
            if alpha > 1:
                sigma_value_label = f'1/{alpha_value_label}'
            else:
                sigma_value_label = f'{1 / alpha}'.rstrip('0').rstrip('.')
            print(f'{fest_label}, alpha = {alpha_value_label}')
            row = 2*(nfest - fi - 1)

            # nalpha, nsnr, ntrains, nfreq = waves.shape
            # nalpha, nsnr, ntrains, nthresh, nfreq = peaks.shape

            # calculate frequency-based estimates
            wave_err = np.zeros((nsnr, ntrains, nks))
            peak_err = np.zeros((nsnr, ntrains, nthresh))
            for snri in range(nsnr):
                for i in range(ntrains):
                    for j in range(nks):
                        wave_err[snri, i, j] = fest_func(waves[ai, snri, i, j])
                    for j in range(nthresh):
                        h = peaks[ai, snri, i, j]
                        if np.sum(h) > 0:
                            peak_err[snri, i, j] = fest_func(h)
                        else:
                            peak_err[snri, i, j] = np.nan

            # update estimates to represent the error
            true_log2_freq = fest_func(train_freq_hists[ai])
            wave_err -= true_log2_freq
            peak_err -= true_log2_freq

            wave_se = wave_err ** 2
            peak_se = peak_err ** 2

            if ai == 0:
                y0 = ys[row]
                y1 = ys[row+1] + hs[row+1]
                fig.text(0.02, 0.5 * (y0 + y1), f'{fest_label}', rotation='vertical',
                         ha='center', va='center', fontsize=14)

            # -----------------------------------------------------------------
            #   Raw RMS

            # setup axes
            ax = fig.add_subplot(position=[xs[ai], ys[row+1], ws[ai], hs[row+1]], sharey=ax_raw)
            ax_raw = ax
            if fi == 0:
                ax.set_title(f'$\\sigma$ = {sigma_value_label}')
            ax.set_yscale('log')
            ax.set_xscale('symlog', linthreshx=snrs[1], linscalex=0.3)
            ax.set_xticks([])
            ax.set_xticks([], minor=True)
            ax.set_xlim(-0.06, 12)
            ax.set_yticks([0.01, 0.1, 1])
            ax.set_yticklabels(['0.01', '0.1', '1'])
            if ai == 0:
                ax.set_ylabel('RMS')
            else:
                axs.append(ax)
            ax.axvline(0, c='0.7', lw=1, ls=':')
            for snr in [0, 0.1, 0.3, 1, 3]:
                ax.axvline(snr, c='0.7', lw=1, ls=':')

            for j in range(nks):
                # plot wave
                ci = np.sqrt(bootci_pi(wave_se[1:, :, j].T, **bootci_kwargs))
                ax.fill_between(snrs[1:], ci[0], ci[1], label=f'Mesaclip ($k$={ks[j]})', **wave_kwargs[j])

                # plot wave snr 0
                ci = np.sqrt(bootci_pi(wave_se[0, :, j], **bootci_kwargs))
                ax.fill_between([-0.03, 0.03], ci[0], ci[1], **wave_kwargs[j])

            for j in range(nthresh):

                # plot peak
                ci = np.sqrt(bootci_pi(peak_se[1:, :, j].T, **bootci_kwargs))
                ax.fill_between(snrs[1:], ci[0], ci[1], label=f'Peak ($\\theta$={threshes[j]})', **peak_kwargs[j])

                # plot peak snr0
                ci = np.sqrt(bootci_pi(peak_se[0, :, j], **bootci_kwargs))
                ax.fill_between([-0.03, 0.03], ci[0], ci[1], **peak_kwargs[j])

            if ai == 0 and fi == 0:
                handles, labels = ax.get_legend_handles_labels()
                handles = list(np.array(handles).reshape(2, -1).T.flatten())
                labels = list(np.array(labels).reshape(2, -1).T.flatten())
                ax.legend(handles, labels, loc='upper left', ncol=3, bbox_to_anchor=(-0.05, 1.75))

            # -----------------------------------------------------------------
            #   Normalised RMS

            # setup axes
            ax = fig.add_subplot(position=[xs[ai], ys[row+0], ws[ai], hs[row+0]], sharey=ax_nrm)
            ax_nrm = ax
            ax.set_yscale('log')
            ax.set_xscale('symlog', linthreshx=snrs[1], linscalex=0.3)
            ax.set_xticks([0, 0.1, 1, 10])
            ax.set_xticks([], minor=True)
            ax.set_xticklabels(['0', '0.1', '1', '10'])
            ax.get_xticklabels()[1].set_ha('left')
            ax.get_xticklabels()[-1].set_ha('right')
            if fi == nfest - 1:
                ax.set_xlabel('SNR')
            ax.set_xlim(-0.06, 12)
            ax.set_yticks([0.01, 0.1, 1])
            ax.set_yticklabels(['0.01', '0.1', '1'])
            if ai == 0:
                ax.set_ylabel('RMS / RMS$_0$')
            else:
                axs.append(ax)
            ax.axhline(1, c='0.7', ls='--', lw=1, zorder=10)
            ax.axvline(0, c='0.7', lw=1, ls=':', zorder=10)
            for snr in [0, 0.1, 0.3, 1, 3]:
                ax.axvline(snr, c='0.7', lw=1, ls=':')

            # plot wave
            for j in range(nks):
                rms0 = np.sqrt(np.nanmean(wave_se[0, :, j]))
                ci = np.sqrt(bootci_pi(wave_se[1:, :, j].T, **bootci_kwargs)) / rms0
                ax.fill_between(snrs[1:], ci[0], ci[1], **wave_kwargs[j])

            # plot peak
            for j in range(nthresh):
                rms0 = np.nanmean(peak_se[0, :, j])
                ci = np.sqrt(bootci_pi(peak_se[1:, :, j].T, **bootci_kwargs)) / rms0
                ax.fill_between(snrs[1:], ci[0], ci[1], **peak_kwargs[j])

        for ax in axs:
            for tk in ax.get_yticklabels():
                tk.set_visible(False)

    fig.savefig('../output/snr_vs_peak_detect_errors.png', dpi=600)
    plt.close(fig)


def main_examples(ntrains):

    alphas, threshes, ks, snrs, freqs, t, base_frequency, \
      train_freq_hists, example_trains, example_signals, waves, peaks = get_results(ntrains)

    # remove the alpha=0.5, doesn't really add anything
    alphas = alphas[1:]
    waves = waves[1:, ...]
    peaks = peaks[1:, ...]
    example_trains = example_trains[1:, ...]
    example_signals = example_signals[1:, ...]
    train_freq_hists = train_freq_hists[1:, ...]

    nalpha, nsnr, ntrains, nthresh, nfreq = peaks.shape
    nks = len(ks)
    log2_freqs = np.log2(freqs)

    np.random.seed(2002171330)
    bootci_kwargs = dict(statfunc=lambda _x: np.nanmean(_x, axis=0), alpha=0.05, n_samples=1000)

    snr_example_idxs = [0]
    for s in [0.1, 0.3, 1, 3]:
        snr_example_idxs.append(np.argmin(np.abs(snrs - s)))

    log2_freq_edges = utils.make_edges(log2_freqs)
    freq_ticks = [1, 2, 4, 8, 16, 32, 64]
    freq_ticklabels = freq_ticks

    wave_kwargs = [dict(facecolor=c, edgecolor=c, alpha=0.5, zorder=100-ci)
                   for ci, c in enumerate(['0', '0.3', '0.5'])]
    peak_kwargs = [dict(facecolor=c, edgecolor=c, alpha=0.5, zorder=90-ci)
                   for ci, c in enumerate(plt.rcParams['axes.prop_cycle'].by_key()['color'][:nthresh])]

    # axes widths and x-positions (indexed from left)
    left_margin = 0.5
    right_margin = 0.1
    column_margin = 0.15
    width_ratios = [left_margin] + reduce((lambda a, b: a+[column_margin]+b), [[1]]*nalpha) + [right_margin]
    xs, ws = ratios_to_pos_and_size(width_ratios)

    # axes heights and y-positions (indexed from bottom)
    top_margin = 0.8
    bottom_margin = 0.6
    height_ratios = [[1.0]]*len(snr_example_idxs) + [[1.5], [0.5], [0.5]]
    height_ratios = [bottom_margin] + reduce((lambda a, b: a+[0.1]+b), height_ratios) + [top_margin]
    height_ratios[-3] = 0.8
    height_ratios[-5] = 0.6
    height_ratios[-7] = 1.1
    ys, hs = ratios_to_pos_and_size(height_ratios)

    fig = plt.figure(figsize=(9, 12))
    fig.text(0.5, 0.98, 'Exploration of noise and irregularity', fontsize=16, ha='center', va='center')
    for ai, alpha in enumerate(alphas):

        alpha_value_label = f'{alpha}'.rstrip('0').rstrip('.')
        if alpha > 1:
            sigma_value_label = f'1/{alpha_value_label}'
        else:
            sigma_value_label = f'{1/alpha}'.rstrip('0').rstrip('.')
        print(f'alpha = {alpha_value_label} ({ai+1}/{nalpha})')

        # -----------------------------------------------------------------
        #   1/ISI histogram

        print('  histogram')

        # plot
        ax_isi = fig.add_subplot(position=[xs[ai], ys[-1], ws[ai], hs[-1]])
        ax_isi.bar(log2_freq_edges[:-1], train_freq_hists[ai], width=np.diff(log2_freq_edges), align='edge', color='k')

        # configure axes
        ax_isi.set_xticks([], minor=True)
        ax_isi.set_xticks(np.log2(freq_ticks))
        ax_isi.set_xticklabels(freq_ticklabels)
        ax_isi.set_title(f'$\\sigma$ = {sigma_value_label}')
        for spine in ['left', 'top', 'right']:
            ax_isi.spines[spine].set_visible(False)
        ax_isi.set_yticks([])
        if ai == 0:
            ax_isi.set_ylabel('True\ndistribution', labelpad=10)
        ax_isi.set_xlabel('Frequency (ISI$^{-1}$)')
        ax_isi.set_xlim([log2_freq_edges[0], log2_freq_edges[-1]])

        # -----------------------------------------------------------------
        #   Raster and signal examples

        print('  raster and signal examples')

        # plot raster
        ax_r = fig.add_subplot(position=[xs[ai], ys[-2], ws[ai], hs[-2]])
        for i in range(example_trains.shape[1]):
            train = example_trains[ai, i]
            spikes = t[np.where(train > 0)[0]]
            ax_r.scatter(spikes, [i] * len(spikes), marker='.', color='k', s=1)

        # plot noisy signal
        ax_s = fig.add_subplot(position=[xs[ai], ys[-3], ws[ai], hs[-3]])
        for y, snri in enumerate(snr_example_idxs):
            signal = example_signals[ai, snri]
            signal = (signal - signal.mean()) / (0.3 + signal.std())  # wierd scaling for visual aesthetic
            ax_s.plot(t, signal + 4*y, c='k', lw=1)

        # configure axes
        if ai == 0:
            ax_r.set_ylabel('True\nraster', labelpad=10)
            ax_s.set_ylabel('SNR examples')
        for ax in [ax_r, ax_s]:
            ax.set_xlim([0, 1])
            for spine in ['left', 'top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.set_yticks([])
            ax.set_xlabel('Time')
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels([f'{tick:g}' for tick in ax.get_xticks()])
        if ai == 0:
            ax_s.set_yticks(4 * np.arange(len(snr_example_idxs)))
            ax_s.set_yticklabels([f'{s:.2g}' for s in snrs[snr_example_idxs]])
            ax_s.yaxis.set_tick_params(length=0)

        # -----------------------------------------------------------------
        #   Wavelet and peak examples

        print('  wavelet and peak examples')

        for i, snr_idx in enumerate(snr_example_idxs):
            ax = fig.add_subplot(position=[xs[ai], ys[i], ws[ai], hs[i]])

            snr_label = f'{snrs[snr_idx]:.2g}'
            print(f'    example snr {snr_label} ({i+1}/{len(snr_example_idxs)})')

            # nalpha, nsnr, ntrains, nthresh, nfreq = peaks.shape
            # nalpha, nsnr, ntrains, nks    , nfreq = waves.shape

            # plot wavelet
            for j in range(nks):
                y = waves[ai, snr_idx, :, j]
                y = y / np.sum(y, axis=-1, keepdims=True)
                ci = np.sqrt(bootci_pi(y, **bootci_kwargs))
                ax.fill_between(log2_freqs, ci[0], ci[1], label=f'Mesaclip ($k$={ks[j]})', **wave_kwargs[j])

            # plot peak
            for j in range(nthresh):
                y = peaks[ai, snr_idx, :, j]
                y = y / np.sum(y, axis=-1, keepdims=True)
                ci = np.sqrt(bootci_pi(y, **bootci_kwargs))
                ax.fill_between(log2_freqs, ci[0], ci[1], label=f'Peak ($\\theta$={threshes[j]})', **peak_kwargs[j])

            # configure axes
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlim(log2_freq_edges[0], log2_freq_edges[-1])
            if ai == 0:
                ax.set_ylabel(f'SNR\n{snr_label}', fontsize=10, labelpad=10)
            if i == 0:
                ax.set_xticks(np.log2(freq_ticks))
                ax.set_xticklabels(freq_ticklabels)
                ax.set_xlabel('Frequency')

        if ai == 0:
            handles, labels = ax.get_legend_handles_labels()
            handles = list(np.array(handles).reshape(2, -1).T.flatten())
            labels = list(np.array(labels).reshape(2, -1).T.flatten())
            ax.legend(handles, labels, loc='upper left', ncol=3, bbox_to_anchor=(-0.05, 1.6))

    fig.savefig('../output/snr_vs_peak_detect_examples.png', dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main_examples(1000)
    # main_errors(1000)
