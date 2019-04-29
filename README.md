An O(n) algorithm for clipping wide peaks from 1D signals.

Given a signal with amplitudes `y[0..n-1]` and monotonically increasing locations `x[0..n-1]`,
clips all peaks that are too wide `x[b] - x[a] > k` for all `b > a` and given parameter `k`.

The following animation shows an example with a uniformally increasing `x` along the horizontal, `y` along the vertical, and `k = 12`:

![algorithm visualisation](output/mesaclip.gif)


The `mesaclip.py` file contains the main algorithm in the `mesaclip` function, and `demo.py` will reproduce the animation.
Running the `mesaclip.py` file will run tests on random signals.
