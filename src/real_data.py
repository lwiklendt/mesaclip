from functools import reduce
import os
import pickle
import numpy as np
import scipy.ndimage
import scipy.stats
import matplotlib.pyplot as plt

import wavelet
import utils


def ratios_to_pos_and_size(ratios):
    s = np.array(ratios) / np.sum(ratios)
    return np.cumsum(s)[:-1:2], s[1::2]


def get_results():

    cache_filename = '../output/real_data_cache.pkl'

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            result = pickle.load(f)

    else:

        xs = np.loadtxt('../data/real_emgs.txt')
        t = np.arange(xs.shape[1]) / 100  # data in 100Hz sampling rate
        n = xs.shape[0]

        freqs = np.geomspace(1/8, 8, 100)
        log2_freqs = np.log2(freqs)
        log2_freq_edges = utils.make_edges(log2_freqs)

        dt = t[1] - t[0]
        scales = np.geomspace(2 * dt, 0.5 * (t[-1] - t[0]), 1000)
        mother = wavelet.Morse(beta=1.58174, gam=3)
        min_cycles = [2, 4, 8]

        threshes = np.linspace(0, 1, 50)
        peak_hists = np.zeros((n, len(threshes), len(freqs)))
        amps = np.zeros((n, len(min_cycles), len(freqs)))

        for i in range(n):

            print(f'signal {i + 1}/{n}')

            x = xs[i]
            x = x - scipy.ndimage.gaussian_filter1d(x, 2/dt)  # subtract baseline
            x = x - np.mean(x)
            x = x / np.max(x)
            xs[i, :] = x

            # peak detect on x
            peak_hist = np.zeros((len(threshes), len(freqs)))
            for j, thresh in enumerate(threshes):
                spikes = utils.spike_detect(x, t, thresh)
                if len(spikes) > 1:
                    f = np.log2(1/np.diff(spikes))
                    h = np.histogram(f, bins=log2_freq_edges)[0].astype(float)

                    if np.max(h) > 0:
                        h = scipy.ndimage.gaussian_filter1d(h, 2)
                        h = h / np.max(h)
                        peak_hist[j, :] = h

            peak_hists[i, ...] = peak_hist

            # wavelet
            for j, k in enumerate(min_cycles):
                w = wavelet.cwt(x, dt, scales, mother, syncsqz_freqs=freqs, min_cycles=k)[0]
                amp = np.mean(np.abs(w)**2, axis=1)
                amp = scipy.ndimage.gaussian_filter1d(amp, 2)
                amps[i, j, :] = np.sqrt(amp)

        result = (xs, t, freqs, threshes, peak_hists, amps)
        with open(cache_filename, 'wb') as f:
            pickle.dump(result, f, protocol=2)

    return result


def main():

    xs, t, freqs, threshes, peak_hists, amps = get_results()

    # subset results for clearer/larger view (using original a..u labels)
    idxs = [ord(c) - ord('a') for c in ['a', 'b', 'c', 'e', 'k', 'm', 'o', 'q', 'u']]
    xs = xs[idxs]
    peak_hists = peak_hists[idxs]
    amps = amps[idxs]

    print(f'delta logHz = {np.log2(freqs[1]) - np.log2(freqs[0]):.3f}')

    n = xs.shape[0]
    nrows = n//2+1

    # setup axis properties
    log2_freqs = np.log2(freqs)
    log2_freq_edges = utils.make_edges(log2_freqs)
    freq_ticks = np.log2([1/8, 1/4, 1/2, 1, 2, 4, 8])
    freq_ticklabels = ['$^1$/$_8$', '$^1$/$_4$', '$^1$/$_2$', '1', '2', '4', '8']
    thresh_edges = utils.make_edges(threshes)
    gridx, gridy = np.meshgrid(log2_freq_edges, thresh_edges)

    # axes widths and x-positions
    left_margin = 0.25
    right_margin = 0.1
    column_margin = 0.3
    width_ratios = [[1.5], [1]]
    width_ratios = [reduce((lambda x, y: x + [0.1] + y), width_ratios)] * 2
    width_ratios = [left_margin] + reduce((lambda x, y: x + [column_margin] + y), width_ratios) + [right_margin]
    ax_xs, ax_ws = ratios_to_pos_and_size(width_ratios)

    # axes heights and y-positions
    bottom_margin = 0.5
    top_margin = 0.1
    row_margin = 0.3
    height_ratios = [[0.5], [1]]
    height_ratios = [reduce((lambda x, y: x + [0.1] + y), height_ratios)] * nrows
    height_ratios = [bottom_margin] + reduce((lambda x, y: x + [row_margin] + y), height_ratios) + [top_margin]
    ax_ys, ax_hs = ratios_to_pos_and_size(height_ratios)

    # flip vertically since we will refer to rows from the top
    ax_ys = ax_ys[::-1]
    ax_hs = ax_hs[::-1]

    fig = plt.figure(figsize=(8.27, 10))

    spine_color = '0.8'

    # draw legend axes
    ax_x = fig.add_subplot(position=[ax_xs[0], ax_ys[1], ax_ws[0], ax_hs[0] + ax_ys[0] - ax_ys[1]])
    ax_p = fig.add_subplot(position=[ax_xs[1], ax_ys[0], ax_ws[1], ax_hs[0]])
    ax_w = fig.add_subplot(position=[ax_xs[1], ax_ys[1], ax_ws[1], ax_hs[1]])
    for ax in [ax_x, ax_w, ax_p]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        for spine in ax.spines:
            ax.spines[spine].set_color(spine_color)
    ax_x.text(0.5, 0.5, 'EMG Signal', ha='center', va='center', fontsize=14)
    ax_w.text(0.5, 0.5, 'Mesaclip\n(our method)', ha='center', va='center', fontsize=10)
    ax_p.text(0.5, 0.5, 'Inverse ISI\nvia peak-detection', ha='center', va='center', fontsize=10)

    threshes_hline = [0.3, 0.5, 0.7]
    threshes_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:3]
    thresh_kwargs = [dict(y=threshes_hline[ci], c=c, lw=2, zorder=50, alpha=0.5)
                     for ci, c in enumerate(threshes_colors)]

    for i in range(n):

        row = (i+1) % nrows
        col_offset = (i+1) // nrows

        # create axes
        r = 2 * row
        c = 2 * col_offset
        ax_x = fig.add_subplot(position=[ax_xs[c+0], ax_ys[r+1], ax_ws[c+0], ax_hs[r+0] + ax_ys[r+0] - ax_ys[r+1]])
        ax_w = fig.add_subplot(position=[ax_xs[c+1], ax_ys[r+1], ax_ws[c+1], ax_hs[r+1]])
        ax_p = fig.add_subplot(position=[ax_xs[c+1], ax_ys[r+0], ax_ws[c+1], ax_hs[r+0]])
        axs = [ax_x, ax_p, ax_w]
        ax_fs = [ax_p, ax_w]

        # plot signals
        ax_x.plot(t, xs[i], lw=1, c='k', zorder=10)
        ax_x.set_ylim(-0.62, thresh_edges[-1])  # hacky manual y-alignment of ax_x and ax_p

        # plot peak frequency estimate
        ax_p.pcolormesh(gridx, gridy, peak_hists[i], cmap='Greys', vmin=-0.2, zorder=10)

        # annotate thresh lines
        for kwargs in thresh_kwargs:
            ax_x.axhline(**kwargs)
            ax_p.axhline(**kwargs)

        # plot wavelet
        ax_w.fill_between(log2_freqs, amps[i, 0]**2, facecolor='k', edgecolor='none')
        # for j in range(amps.shape[1]):
        #     amp = amps[i, j]**2
        #     amp = amp / np.sum(amp)
        #     ax_w.plot(log2_freqs, amp, alpha=0.7)

        # axes refinement
        ax_x.set_ylabel(f'({chr(ord("a") + i)})',
                        position=(0, 0.94), rotation='horizontal', va='center', fontsize=12, labelpad=12)
        for ax in axs:
            ax.set_yticks([])
            for spine in ax.spines:
                ax.spines[spine].set_color(spine_color)
        if row == nrows - 1:
            ax_x.set_xlabel('Time (s)')
            ax_w.set_xlabel('Frequency (Hz)')
            ax_x.set_xticks(range(0, 11, 2))
            ax_w.set_xticklabels(freq_ticklabels)
            for ax in axs:
                ax.spines['bottom'].set_visible(True)
        else:
            ax_x.set_xticks([])
            ax_w.set_xticklabels([])
        ax_p.set_xticklabels([])
        for ax in ax_fs:
            ax.spines['bottom'].set_visible(True)
            ax.set_xlim([log2_freq_edges[0], log2_freq_edges[-1]])
            ax.tick_params(axis='x', which='major', length=2)
            ax.set_xticks(freq_ticks)

    fig.savefig('../output/real_data.png', dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main()
