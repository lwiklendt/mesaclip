import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import scipy.ndimage
import scipy.signal

import colorednoise

import wavelet
import utils


def plot_mesaclipped(t, x, fs, mother1, mother2, freqs, sync_sqz, fig_filename, snr=np.inf, noise_gamma=0, num_scales=2000, split=False):

    print(f'plotting {fig_filename}')

    # add noise
    noise = colorednoise.powerlaw_psd_gaussian(noise_gamma, len(t))
    x = x + np.sqrt(np.var(x) / snr) * noise

    dt = t[1] - t[0]

    min_cycles = 2

    if sync_sqz:
        syncsqz_freqs = freqs
    else:
        syncsqz_freqs = None

    # compute wavelet transforms
    amps = []
    mus = []
    vmax = -np.inf
    xr = []  # for reconstructed signals
    cois = []
    titles = []
    # for i, (mother, c) in enumerate(zip([mother1, mother1, mother2, mother2], [0, 1, 0, 1])):
    for i, (mother, c, title) in enumerate(zip([mother1, mother2], [0, 1], ['Conventional', 'MesaClip'])):

        if sync_sqz:
            scales = np.geomspace(0.5 * dt, t[-1] - t[0], num_scales)
        else:
            scales = mother.convert_freq_scale(freqs)

        if split:
            x_pos = x.copy()
            x_neg = (-x).copy()
            x_pos[x_pos < 0] = 0
            x_neg[x_neg < 0] = 0
            w_pos, _, coi_time_idxs, coi_freq_idxs = wavelet.cwt(x_pos, dt, scales, mother, syncsqz_freqs=syncsqz_freqs, min_cycles=c * min_cycles)
            w_neg, _, coi_time_idxs, coi_freq_idxs = wavelet.cwt(x_neg, dt, scales, mother, syncsqz_freqs=syncsqz_freqs, min_cycles=c * min_cycles)
            phi_pos = np.unwrap(np.angle(w_pos))
            phi_neg = np.unwrap(np.angle(w_neg))
            w = (np.abs(w_pos) + np.abs(w_neg)) * np.exp(0.5j * (phi_pos + phi_neg))
        else:
            w, _, coi_time_idxs, coi_freq_idxs = wavelet.cwt(x, dt, scales, mother, syncsqz_freqs=syncsqz_freqs, min_cycles=c * min_cycles)

        coi_times = t[coi_time_idxs] - t[0]
        coi_freqs = freqs[coi_freq_idxs]
        cois.append((coi_times, coi_freqs))

        amp = np.abs(w)
        mus.append(np.sqrt(np.mean(amp ** 2, axis=1)))

        if sync_sqz:
            amp = scipy.ndimage.gaussian_filter(amp, [2, 2])  # smooth a bit so that sharp bits show up in the high-res

        amps.append(amp)
        vmax = max(vmax, np.max(amp))
        xr.append(wavelet.reconstruct(w, mother, scales))

        titles.append(title)

    nw = len(amps)

    # create figure and axes
    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(1 + nw, 2, width_ratios=[1, 0.2], height_ratios=[1, ] + [2, ] * nw)
    ax = np.zeros((1 + nw, 2), dtype=object)
    ax[0, 0] = fig.add_subplot(gs[0, 0])  # signal
    ax_s = None
    for i in range(1, 1 + nw):            # wavelet transform results
        ax[i, 0] = fig.add_subplot(gs[i, 0])
        ax[i, 1] = fig.add_subplot(gs[i, 1], sharex=ax_s)
        ax_s = ax[i, 1]

    # convenience axes variables
    ax_sig = ax[0, 0]
    ax_w = ax[1:, 0]
    ax_s = ax[1:, 1]

    # plot signal
    ax_sig.plot(t, x, c='k', lw=0.5)
    ax_sig.tick_params(which='both', direction='out')
    ax_sig.set_xlim(t[0], t[-1])
    ax_sig.set_xticklabels([])
    ax_sig.set_yticks([])
    ax_sig.xaxis.set_ticks_position('bottom')
    ax_sig.yaxis.set_ticks_position('left')
    ax_sig.axis('off')
    ax_sig.set_title('Signal', loc='left')

    # pcolormesh grid coordinates
    t_edges = utils.make_edges(t)
    f_edges = utils.make_edges(freqs, log=True)
    t_grid, f_grid = np.meshgrid(t_edges, f_edges)

    # cmap = 'gray_r'
    # cmap = 'binary'
    # cmap = 'bone_r'
    cmap = 'Blues'

    # iterate over each amplitude type
    for pi, (mu, amp, (coi_times, coi_freqs), title) in enumerate(zip(mus, amps, cois, titles)):

        # plot amplitudes
        vmax = np.max(amp)
        # vmin = -0.05 * vmax  # when using a white to color colormap, make backgroud slightly off-white
        vmin = 0
        ax_w[pi].pcolormesh(t_grid, f_grid, amp, cmap=cmap, rasterized=True, vmin=vmin, vmax=vmax)
        ax_w[pi].set_xlim(t[0], t[-1])

        ax_w[pi].set_title(title, loc='left')

        # # plot cone-of-influence
        # coi_kwargs = dict(c='k', ls='--', lw=0.5, alpha=0.5)
        # ax_w[pi].plot(t[ 0] + coi_times, coi_freqs, **coi_kwargs)
        # ax_w[pi].plot(t[-1] - coi_times, coi_freqs, **coi_kwargs)

        # plot time-averages
        ax_s[pi].fill(np.r_[0, mu, 0], np.r_[freqs[0], freqs, freqs[-1]],
                      c='k', lw=0, zorder=2, alpha=0.2)
        ax_s[pi].plot(mu, freqs, c='k', lw=1, zorder=3)

        # setup axes ranges and ticks
        for ax_ws in [ax_w[pi], ax_s[pi]]:
            ax_ws.tick_params(which='both', direction='out')
            ax_ws.xaxis.set_ticks_position('bottom')
            ax_ws.yaxis.set_ticks_position('left')
            ax_ws.set_yscale('log')
            ax_ws.set_ylim(freqs[0], freqs[-1])

        for f in fs:
            ax_s[pi].axhline(f, lw=1, ls=':', color='r', alpha=1.0, zorder=5)

        ax_w[pi].set_ylabel(r'Freq (Hz)')
        ax_w[pi].set_xlim(ax[0, 0].get_xlim())

        # clean up signal axes
        ax_s[pi].set_yticklabels([])
        ax_s[pi].set_xticks([])
        for side in ['top', 'right', 'bottom']:
            ax_s[pi].spines[side].set_visible(False)

    for i in range(len(ax_w) - 1):
        ax_w[i].set_xticklabels([])

    ax[-1, 0].set_xlabel(r'Time (s)')

    fig.tight_layout(h_pad=0.2, w_pad=0.5, rect=[-0.025, -0.03, 1.025, 0.98])
    fig.savefig(fig_filename, dpi=300)


def chirp_thresh_data(nsamp):

    # t = np.linspace(0, 20, nsamp) - 5
    t = np.linspace(-1, 11, nsamp)

    fs = [1, 10]
    x = 1.0 * (scipy.signal.chirp(t, fs[0], fs[1], 10) > 0.9)
    x[(t < -0.1) | (t > 10)] = 0

    return t, x, fs


def burst_dirac(nsamp):
    t = np.linspace(-1, 11, nsamp)

    fs = [0.5, 5]

    # dirac comb
    ds = int(round(1/(fs[1]*(t[1] - t[0]))))
    x = np.zeros(nsamp)
    x[ds//4::ds] = 1

    # remove parts of comb to generate burst-like behaviour
    x *= (np.sin(fs[0] * 2 * np.pi * (t + 0.5)) > 0)

    return t, x, fs


def dirac_from_sin(nsamp):
    t = np.linspace(-1.5, 11.5, nsamp)
    fs = [1]
    x = 0.5 * np.sin(2 * np.pi * fs[0] * (t + fs[0] * 0.25)) + 0.5
    x = x ** np.geomspace(1, 100, nsamp)
    x = x * np.linspace(1, 10, nsamp)
    return t, x, fs


def poisson_data(nsamp, freqs):
    t = np.linspace(-1, 11, nsamp)
    x = np.zeros_like(t)

    rate = 5
    scale = 1 / rate

    # log-exponential pdf (using change of variables for the exponential distribution)

    def f_x(var_x):
        return scale * np.exp(-scale * var_x)

    g = np.log
    g_inv = np.exp

    def f_y(var_y):
        return f_x(g_inv(var_y)) * g_inv(var_y)

    fs = f_y(g(freqs))

    # generate spike train
    np.random.seed(4)
    last_t = 2 * t[0] - t[-1]  # start double the time range back to burn in the spike train sample
    while last_t < t[-1]:
        last_t += np.random.exponential(scale=scale)
        if t[0] <= last_t <= t[-1]:
            x[np.searchsorted(t, last_t)] = 2.5
    x = scipy.ndimage.gaussian_filter1d(x, sigma=1)

    return t, x, fs


def main_artificial():

    nfreq = 300
    sync_sqz = True
    nsamp = 4096

    min_freq = 10 ** -1
    max_freq = 10 ** 2
    freqs = np.geomspace(min_freq, max_freq, nfreq, endpoint=True)

    mother1 = wavelet.Morse(beta=12, gam=3)
    mother2 = wavelet.Morse(beta=1.58174, gam=3)

    data_funcs = [
        (burst_dirac, 'burst_dirac'),
        (dirac_from_sin, 'dirac_from_sin'),
        (chirp_thresh_data, 'chirp_thresh'),
    ]
    for func, fname in data_funcs:
        t, x, fs = func(nsamp)
        # plot_mesaclipped(t, x, fs, mother1, mother2, freqs, sync_sqz, f'../output/{fname}.pdf')
        plot_mesaclipped(t, x, fs, mother1, mother2, freqs, sync_sqz, f'../output/{fname}.png')
        # pulse_filtered(t, x, fs, mother1, mother2, freqs, sync_sqz, f'../output/{fname}_noise.png', snr=1)


def main_real():
    x = np.loadtxt('../data/real_emg.txt', skiprows=100)

    x = x[650:-4150, :]
    t = x[:, 0]
    x = x[:, 1]

    t = t - t[0]
    x = x - scipy.ndimage.gaussian_filter1d(x, 100)

    spikes = np.where(x < -3)[0]
    mean_spike_freq = 1 / np.mean(np.diff(t[spikes]))
    fs = [mean_spike_freq]

    nfreq = 300
    sync_sqz = True

    min_freq = 10 ** -1
    max_freq = 10 ** 2
    freqs = np.geomspace(min_freq, max_freq, nfreq, endpoint=True)

    mother1 = wavelet.Morse(beta=12, gam=3)
    mother2 = wavelet.Morse(beta=1.58174, gam=3)

    # plot_mesaclipped(t, x, fs, mother1, mother2, freqs, sync_sqz, f'../output/real.pdf', num_scales=1000, split=True)
    plot_mesaclipped(t, x, fs, mother1, mother2, freqs, sync_sqz, f'../output/real.png', num_scales=1000, split=True)


if __name__ == '__main__':
    main_artificial()
    main_real()
