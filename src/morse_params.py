import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
from matplotlib import gridspec

import wavelet


def mid_amp(beta, gamma, omega):

    # create frequency-domain wavelet to capture first harmonic

    # peak_hz = (beta/gamma)**(1/gamma) / (2 * np.pi)
    # s = peak_hz / 2
    # a = 2 * (np.e * gamma / beta)**(beta/gamma)
    # psi_fspace = a * ((s * omega) ** beta) * np.exp(-(s * omega) ** gamma)

    mother = wavelet.Morse(beta=beta, gam=gamma)
    s = mother.convert_freq_scale(2)
    psi_fspace = mother.freq_domain(s * omega)

    # wavelet transform of Dirac comb
    x = np.fft.ifft(np.sqrt(s) * psi_fspace)

    # half-way between two diracs as ratio between 1st harmonic and fundamental
    a = abs(x[len(x)//2])

    print(f'  eval at beta={beta:.8f}: amp={a:g}', flush=True)

    return a


def calc_optimum_beta():
    omega = 2 * np.pi * np.arange(0, 2**16)
    beta_opt, min_amp, _, _ = fminbound(lambda beta: mid_amp(beta, gamma=3, omega=omega), 0.5, 4, xtol=1e-10, full_output=True)
    print(f'optimal beta={beta_opt:.8f} with amp={min_amp:g}')
    return beta_opt


def main():

    beta_opt = calc_optimum_beta()
    gamma = 3

    n = 2**12
    freq = np.arange(0, n)
    omega = freq * 2 * np.pi

    nb = 2**10
    betas = np.geomspace(0.5, 16, nb)
    amps = np.zeros(nb)

    for i, beta in enumerate(betas):

        mother = wavelet.Morse(beta=beta, gam=gamma)
        s = mother.convert_freq_scale(2)
        psi_fspace = mother.freq_domain(s * omega)

        # wavelet transform of Dirac comb
        x = np.fft.ifft(n * np.sqrt(s) * psi_fspace)

        # half-way between two diracs as ratio between 1st harmonic and fundamental
        a = abs(x[len(x)//2])

        amps[i] = a

    betas_plot = [0.5, beta_opt, 4, 12]

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, len(betas_plot), height_ratios=[1.5, 1])

    # plot amplitudes at varying betas
    ax = fig.add_subplot(gs[0, :])
    ax.plot(betas, amps, c='k', lw=2)
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('Amplitude')
    ax.set_title('Amplitude at 1st harmonic halfway between two Dirac delta functions')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.5, 1, beta_opt, 2, 4, 8, 16])
    ax.set_xticklabels(['0.5', '1', '$\\beta*$', '2', '4', '8', '16'])
    ax.set_xticks([], minor=True)

    t = np.linspace(-10, 10, 2**10)
    dt = t[1] - t[0]
    x = np.zeros_like(t)
    x[len(x)//2] = 1

    # plot wavelets for a few selected betas
    for i, beta in enumerate(betas_plot):
        ax = fig.add_subplot(gs[1, i], facecolor='0.9')

        if np.abs(beta - beta_opt) < 1e-4:
            ax.set_title(f'$\\beta = \\beta* = {beta:.6g}$')
        else:
            ax.set_title(f'$\\beta = {beta:.6g}$')

        mother = wavelet.Morse(beta=beta, gam=gamma)
        w = wavelet.cwt(x, dt, np.array([1]), mother)[0][0]

        ax.axhline(0, c='k', lw=1, ls='--')
        ax.fill_between(t, -np.abs(w), np.abs(w), color='w', lw=1.5)
        ax.plot(t, np.real(w), lw=1.5)
        ax.plot(t, np.imag(w), lw=1.5)
        ax.set_xlim(t[0], t[-1])

        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig('../output/beta_opt.pdf', dpi=300)
    fig.savefig('../output/beta_opt.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
