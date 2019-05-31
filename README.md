An O(n) algorithm for clipping wide peaks from 1D signals, designed for application to removing harminc artefacts from the continuous wavelet transform.

Given a signal with amplitudes `y[0..n-1]` and non-decreasing locations `x[0..n-1]`,
clips all peaks that are too wide `x[b] - x[a] > k` for all `b > a` and given parameter `k`.

The following animation shows an example with a uniformally increasing `x` along the horizontal, `y` along the vertical, and `k = 12`:

![algorithm visualisation](output/mesaclip.gif)


The `mesaclip.py` file contains the main algorithm in the `mesaclip` function, and `demo.py` will reproduce the animation.
Running the `mesaclip.py` file will run tests on random signals.

The following table shows four examples comparing a convential continuous wavelet transform and one with mesaclipping applied.
The middle subplots use a conventional Morse(&beta;=12, &gamma;=3) wavelet transform, and the bottom subplots use a Morse(&beta;=1.58174, &gamma;=3) wavelet with mesaclipping applied. The mesaclipped version does not suffer from harmonic artefacts generated due to sharp changes in the singals.

| | |
:---:|:---:
From a smooth to a spiky signal ![](output/dirac_from_sin.png) | Bursts of Dirac deltas ![](output/burst_dirac.png)
Thresholded chirp               ![](output/chirp_thresh.png)   | Real EMG signal        ![](output/real.png)
