import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.genfromtxt('zphot.out')
    # dict = ['Identification', 'Best Z', 'Z_best68_low', 'z_best68_high', '']
    dict = ['IDENT', 'Z_BEST', 'Z_BEST68_LOW', 'Z_BEST68_HIGH', 'Z_ML',
            'CHI_BEST', 'MOD_BEST', 'EXTLAW_BEST', 'EBV_BEST', 'PDZ_BEST',
            'SCALE_BEST', 'DIST_MOD_BEST', 'NBAND_USED', 'Z_SEC', 'CHI_SEC',
            'MOD_SEC', 'Z_QSO', 'CHI_QSO', 'MOD_QSO', 'MOD_STAR',
            'CHI_STAR', 'CONTEXT', 'ZSPEC']
    # data_processed = np.array(len(data))
    # print(data_processed)
    z_best = data[:, -1]
    z_best = np.log10(z_best)
    x = z_best
    zspec = data[:, -1]
    zspec = np.log10(zspec)
    y = zspec

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=2500)
    # print(np.max(heatmap))
    # print(xedges)
    print(len(heatmap))
    print(heatmap)
    # heatmap = heatmap[[0,50],[0, 50]]
    # heatmap = heatmap[[1, 3], :][:, [1, 3]]

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # extent = [0, 0.8, 0, 0.8]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower', vmin=0, vmax=5)
    plt.colorbar()
    plt.show()


main()
