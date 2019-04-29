import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as manim

import numpy as np

from mesaclip import find_extrema


def mesaclip_for_vis(x, y, k):

    # ensure monotonically increasing x
    if np.any(np.diff(x) < 0):
        yield ('x is not monotonically increasing', None)
        raise RuntimeError('x is not monotonically increasing')

    # if entire signal is too short, then clip everything
    if x[-1] - x[0] < k:
        y[:] = np.min(y)
        yield ('signal too short', None)
        return

    n = len(y)

    # 1 for peak, -1 for trough, 0 otherwise
    m = find_extrema(y)

    # stack stores active ranges to the left of the current peak, stack size is si, top element is at si - 1
    si = 0
    sa = np.empty(n//2 + 1, dtype=np.int64)  # impossible to have more peaks, and hence ranges, than ceil(n/2)
    sb = np.empty(n//2 + 1, dtype=np.int64)

    # scan for peaks
    i = -1
    while i < n - 1:

        i += 1
        yield ('next i', dict(i=i))

        if m[i] == 1:

            # found peak

            # hold an expanding range (a, b + 1) where a can decrease and b increase,
            a = i
            b = i

            # expanding the range to the left and right, there are only three ways to break out of this loop:
            #   1) when the range is large enough to span the desired minimum distance: x[b] - x[a] >= k
            #   2) can no longer increase the range because a == 0 and b == n - 1
            #   3) we've reached a trough on the right side of the range
            while True:

                yield ('range', dict(range=(a, b)))

                # calculate the range width
                d = x[b] - x[a]

                # if we have sufficient range width
                if d >= k:

                    # clip this range
                    # y[a:b + 1] = min(y[a], y[b])  <- true algorithm, but below shows all indexes
                    for clip_i in range(a, b + 1):
                        y[clip_i] = min(y[a], y[b])
                        yield ('clipping', dict(clip_i=clip_i, range=(a, b)))

                    # go to next peak
                    i = b
                    break

                # we've reached a left trough that isn't at the start
                if m[a] == -1 and a > 0:

                    # can the range on the top of the stack be combined with the current range?
                    if si > 0 and sb[si - 1] == a:

                        # pop and combine
                        si -= 1
                        t = sa[si]
                        y[t] = min(y[t], y[a])  # retain potential minimum at the combine index
                        a = t

                        yield('combining', dict(range=(a, b), r=(si, sa, sb)))

                        # we've just extended the range, so loop around to test against k
                        continue

                    else:
                        # push onto stack
                        sa[si] = a
                        sb[si] = b
                        si += 1

                        yield('push stack', dict(r=(si, sa, sb)))

                        # go to next peak
                        i = b
                        break

                # we've reached a right trough that isn't at the end
                if m[b] == -1 and b < n - 1:

                    # push onto stack
                    sa[si] = a
                    sb[si] = b
                    si += 1

                    yield('push stack', dict(r=(si, sa, sb)))

                    # go to next peak
                    i = b
                    break

                # left or right step forced by range touching an end
                if a == 0:
                    b += 1
                elif b == n - 1:
                    a -= 1

                # otherwise step towards the larger value
                elif y[a - 1] > y[b + 1]:
                    a -= 1
                else:
                    b += 1

    # clip all remaining ranges
    while si > 0:
        si -= 1
        a = sa[si]
        b = sb[si]
        # y[a:b + 1] = min(y[a], y[b])  <- true algorithm, but below shows all indexes
        yield ('clipping', dict(range=(a, b), r=(si, sa, sb)))
        for clip_i in range(a, b + 1):
            y[clip_i] = min(y[a], y[b])
            yield ('clipping', dict(clip_i=clip_i, r=(si, sa, sb), range=(a, b)))

    yield ('done', dict())


def main(movie_type):

    # data
    x = np.arange(47)
    y = np.zeros(47) + 0.1
    ls = [5, 18, 34, 40]
    ws = [2, 7, 2.5, 2.5]
    hs = [0.7, 0.6, 1, 0.8]
    for l, w, h in zip(ls, ws, hs):
        y += h * np.exp(-((x - l) / float(w))**2)
    k = 12

    hx = 0.5 * (x[1] - x[0])

    y_orig = y.copy()

    single_frame_debug = False
    draw_frame_number = False
    draw_k = False
    draw_curr_index = False

    save_frames = [13, 49, 65, 74, 75, 92, 95, 103]

    fig = plt.figure(figsize=(6, 2))
    w, h = 0.98, 0.95
    ax = fig.add_axes((0.5 * (1 - w), 0.05, w, h))

    if not os.path.exists('../output/frames'):
        os.makedirs('../output/frames')

    # create video writer
    if movie_type == 'gif':
        writer = manim.writers['imagemagick'](fps=10)
        writer.setup(fig, '../output/mesaclip.gif')
    elif movie_type == 'mp4':
        writer = manim.writers['ffmpeg'](fps=10, bitrate=8000, codec='mpeg4')
        writer.setup(fig, '../output/mesaclip.mp4', dpi=300)
    else:
        raise RuntimeError(f'unknown movie type: {movie_type}')

    frame_idx = 1

    state = dict()
    ylim = 1.2

    ms_data = 10
    ms_curr = 15

    for curr_state_type, curr_state in mesaclip_for_vis(x, y, k):

        # update state
        for key, value in curr_state.items():
            state[key] = value

        # clear all state data
        if curr_state_type == 'done':
            state = dict()

        ax.clear()

        # plot k
        if draw_k:
            ax.plot(x[:k], ylim + 0 * x[:k], c='k', lw=1, marker='.', ms=ms_data, mec='none', mfc='k',
                    zorder=50, clip_on=False)
            ax.text(x[0], ylim - 0.04, '$k={}$'.format(k), ha='left', va='top')

        # plot data
        ax.fill_between(x, y_orig, edgecolor='none', facecolor='0.85', zorder=1)
        ax.fill_between(x, y, edgecolor='none', facecolor='k', zorder=2, alpha=0.25)
        ax.plot(x, y, c='k', lw=1, marker='.', ms=ms_data, mec='none', mfc='k', zorder=5)

        # plot current index
        if 'i' in curr_state and draw_curr_index:
            i = curr_state['i']
            ml, sl, bl = ax.stem([x[i], ], [y[i], ])
            plt.setp(ml, marker='.', ms=ms_curr, mec='k', mfc='k', mew=1, zorder=10)
            plt.setp(sl, lw=2, c='k', zorder=9)

        # plot current clipping index
        if 'clip_i' in curr_state and draw_curr_index:
            i = curr_state['clip_i']
            ml, sl, bl = ax.stem([x[i], ], [y[i], ])
            plt.setp(ml, marker='.', ms=ms_curr, mec='k', mfc='k', mew=1, zorder=10)
            plt.setp(sl, lw=2, c='k', zorder=9)

        # plot current range
        if 'range' in curr_state:
            i0, i1 = curr_state['range']
            ax.fill_between(x[i0:i1+1], y[i0:i1+1], lw=1, edgecolor='g', facecolor='g', zorder=4)
            ax.plot(x[i0], y[i0], ls='none', marker='.', ms=ms_curr, mec='k', mew=1, mfc=(0, 1, 0), zorder=25)
            ax.plot(x[i1], y[i1], ls='none', marker='.', ms=ms_curr, mec='k', mew=1, mfc=(0, 1, 0), zorder=25)

        # plot stacked ranges
        if 'r' in state:
            ri, r0, r1 = state.get('r')
            while ri > 0:
                ri -= 1
                i0 = r0[ri]
                i1 = r1[ri]
                ax.fill_between(x[i0:i1+1], y[i0:i1+1], lw=1, edgecolor='m', facecolor='m', zorder=3)
                ax.plot(x[i0], y[i0], ls='none', marker='.', ms=ms_curr, mec='k', mew=1, mfc=(1, 0, 1), zorder=20)
                ax.plot(x[i1], y[i1], ls='none', marker='.', ms=ms_curr, mec='k', mew=1, mfc=(1, 0, 1), zorder=20)

        # setup limits and axes aesthetics
        ax.set_ylim(0, ylim)
        ax.set_xlim(x[0] - hx, x[-1] + hx)
        ax.axis('off')

        # debug text for frame number
        if draw_frame_number:
            ax.text(0.5, 1, 'Frame {}'.format(frame_idx), ha='center', va='top', transform=ax.transAxes)

        if frame_idx in save_frames:
            fig.savefig('../output/frames/frame{:03}.pdf'.format(frame_idx))

        # debug single frame for setting up data
        if not single_frame_debug or i == 23:
            writer.grab_frame()
            print('frame {}: {}'.format(frame_idx, curr_state_type))
            frame_idx += 1
            sys.stdout.flush()

    writer.finish()


if __name__ == '__main__':
    # main('gif')
    main('mp4')
