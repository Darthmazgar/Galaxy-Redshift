import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f as f_stats


def read_lephare_data(data='zphot.out'):  # zphot.out, 2SLAQ_Original.txt
    print("*************************************************")
    print("Loading data ... \n")
    d = np.genfromtxt(data)
    print("Loaded.\n")
    z_phot = (d[:, 1])  # for zphot.out
    z_spec = (d[:, -1])
    return z_phot, z_spec

def f_stats(z_phot, z_spec):
    z_phot = bin_data_series(z_phot, bins=0, select_bins=True)
    z_spec = bin_data_series(z_spec, bins=0, select_bins=True)
    plot_densitymap(z_phot, z_spec)


def bias(z_phot, z_spec):
    b = np.zeros(len(z_phot))
    for i in range(len(z_phot)):
        b[i] = z_phot[i] - z_spec[i]
    # print("BIAS DATA:\n ", b)
    return b


def sigma_z(z_phot, z_spec):
    sigma = np.zeros(len(z_phot))
    for i in range(len(z_phot)):
        sigma[i] = np.sqrt((z_phot[i] - z_spec[i])**2)
    return sigma


def sigma_z2(z_phot):
    mean = np.average(z_phot)
    sigma_2 = np.zeros(len(z_phot))
    for i in range(len(z_phot)):
        sigma_2[i] = np.sqrt((z_phot[i] - mean)**2)
    return sigma_2


def sigma_z3(z_spec):
    mean = np.average(z_spec)
    sigma_3 = np.zeros(len(z_spec))
    for i in range(len(z_spec)):
        sigma_3[i] = np.sqrt((z_spec[i] - mean)**2)
    return sigma_3

def plot_sig_photoz(sigma, z_spec, n_data_sets=1):
    # sigma = [sigma for x in range(len(z_phot))]
    plt.plot(z_spec, sigma, '*')
    plt.ylabel("$1\sigma$ scatter around mean spec-z")
    plt.xlabel("Spectroscopic Redshift")
    plt.show()

def plot_densitymap(z_phot, z_spec, div_zero_rem=True, show=True, save=True):
    fig = plt.figure()
    print("*************************************************")
    print("Plotting heatmap ... \n")
    # heatmap, xedges, yedges = np.histogram2d(z_spec, z_phot, bins=200)
    heatmap, xedges, yedges, img = plt.hist2d(z_spec, z_phot, bins=100, range=[[0.25, 0.8], [0.25, 0.8]])

    h_copy = heatmap.copy()
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    extent = [0.25, 0.8, 0.25, 0.8]
    if div_zero_rem:
        for i in range(len(heatmap)):
            for j in range(len(heatmap[i])):
                if heatmap[i][j] == 0:
                    heatmap[i][j] = np.e
    heatmap = -np.log(heatmap)

    if show:
        fig.clear()
        y = np.linspace(0.26, 0.78, 10)
        x = y
        plt.plot(x, y, color=(0,0,0))
        plt.imshow(heatmap.T, origin='lower', interpolation='gaussian', extent=extent) # extent=extent, origin='lower')
        plt.colorbar()
        plt.xlabel("Spectroscopic Redshift")
        plt.ylabel("Le Phare Photo-z")
        if save:
            plt.savefig('C:/Users/iainm/OneDrive/Uni 2018-19/Senior Honours Project/Galaxy-Redshift/Data_analysis/Images/heatmap.jpg')
        plt.show()
    return h_copy, xedges, yedges

def remove_erronious_data(z_spec, z_phot, dl):
    print("*************************************************")
    print("Removing erronious data ... \n")

    index_ar = []
    for i in range(dl):
        if z_spec[i] <= 0.01 or z_phot[i] > 1. :
            index_ar.append(i)
    z_phot = np.delete(z_phot, index_ar)
    z_spec = np.delete(z_spec, index_ar)
    print("%d Data points removed.\n\n" % (dl - len(z_phot)))
    return z_spec, z_phot


def bin_data_series(data, bin_len, name=False, select_bins=False):
    print("*************************************************")
    if name:
        print("Binning data series: %s\n" % name)
    bins = np.linspace(min(data[:,1]), max(data[:,1]), bin_len)
    avg = np.zeros(len(bins))
    sum = np.zeros(len(bins))
    count = np.zeros(len(bins))
    for i in range(len(data[:,0])):
        j = 1
        while data[i, 1] >= bins[j]+0.0001*bins[j]:  # HACKKKEEEE
            j += 1
        sum[j] += data[i, 0]
        count[j] += 1
    # for k in range(len(avg)):
    #     if count[k] != 0:
    #         avg[k] = sum[k] / count[k]
    avg = sum / count
    std = np.zeros(len(avg))
    for i in range(len(std)):
        if count[i] != 0: # and avg[i]**2 <= sum[i]/count[i]:
            # std[i] = np.sqrt(np.abs(sum[i]/count[i]) - (avg[i]**2))
            # std[i] = np.sqrt((sum[i] / count[i] - avg[i])**2)
            # std[i] = np.sqrt(count[i]) * avg[i]
            pass
    return avg, std

def zip_data(data, bin_wrt):
    zipped = np.array(list(zip(data, bin_wrt)))
    zipped = zipped[np.argsort(zipped[:, 1])]
    return zipped

def linspace_data_series(data, bins, name):
    print("*************************************************")
    print("Splitting data series: %s, in to bins\n" % name)
    bin = np.linspace(min(data), max(data), bins)
    return bin

def plot_bias(zipped_bias, binned_bias, lin_space_z_spec, err=0, plot_raw=True, save=True):
    col = (0,0,1)
    if plot_raw:
        plt.plot(zipped_bias[:,1], zipped_bias[:,0], '*', label='Raw data')
        col = (1,0,0)
    plt.plot(lin_space_z_spec[1:-2], binned_bias[1:-1], color=col, linewidth=2, label='Binned average data')
    # plt.errorbar(lin_space_z_spec[1:-2], binned_bias[1:-1], color=col, linewidth=2, yerr=err[1:-1], label='Binned average data')
    if plot_raw:
        plt.legend()
    # plt.title('Bias vs spectrascopic redshift')
    plt.xlabel('Spectrascopic redshift')
    plt.ylabel('Bias')
    if save:
        if plot_raw:
            plt.savefig('C:/Users/iainm/OneDrive/Uni 2018-19/Senior Honours Project/Galaxy-Redshift/Data_analysis/Images/z-phot-bias%s.jpg' % 'w-raw')
        else:
            plt.savefig('C:/Users/iainm/OneDrive/Uni 2018-19/Senior Honours Project/Galaxy-Redshift/Data_analysis/Images/z-phot-bias.jpg')
    plt.show()

def plot_sigma(zipped_sigma, binned_sigma, lin_space_z_spec, i, err=0, plot_raw=True, save=True):
    col = (0,0,1)
    if plot_raw:
        plt.plot(zipped_sigma[:,1], zipped_sigma[:,0], '*', label='Raw data')
        col = (1,0,0)
    plt.plot(lin_space_z_spec[1:-2], binned_sigma[1:-1], color=col, linewidth=2, label='Binned average data')
    # plt.errorbar(lin_space_z_spec[1:-2], binned_sigma[1:-1], color=col, linewidth=2, yerr=err[1:-1], label='Binned average data')
    if plot_raw:
        plt.legend()
    # plt.title('$%d\sigma$ scatter vs spectrascopic redshift' % i)
    plt.xlabel('Spectrascopic redshift')
    plt.ylabel('%d$\sigma$ scatter around photo-z' % i)
    if save:
        if plot_raw:
            plt.savefig('C:/Users/iainm/OneDrive/Uni 2018-19/Senior Honours Project/Galaxy-Redshift/Data_analysis/Images/scatter-about-photo-z%s.jpg' % 'w-raw')
        else:
            plt.savefig('C:/Users/iainm/OneDrive/Uni 2018-19/Senior Honours Project/Galaxy-Redshift/Data_analysis/Images/scatter-about-photo-z.jpg')
    plt.show()

def plot_sigma_mean(zipped_sigma, binned_sigma, lin_space_z_spec, i, err=0, plot_raw=True, save=True):
    col = (0,0,1)
    if plot_raw:
        plt.plot(zipped_sigma[:,1], zipped_sigma[:,0], '*', label='Raw data')
        col=(1,0,0)
    plt.plot(lin_space_z_spec[1:-2], binned_sigma[1:-1], color=col, linewidth=2, label='Binned average data')
    # plt.errorbar(lin_space_z_spec[1:-2], binned_sigma[1:-1], color=col, linewidth=2, yerr=err[1:-1], label='Binned average data')
    if plot_raw:
        plt.legend()
    # plt.title('$%d\sigma$ scatter around the mean photo-z vs spectrascopic redshift' % i)
    plt.xlabel('Spectrascopic redshift')
    plt.ylabel('%d$\sigma$ scatter around mean photo-z' % i)
    if save:
        if plot_raw:
            plt.savefig('C:/Users/iainm/OneDrive/Uni 2018-19/Senior Honours Project/Galaxy-Redshift/Data_analysis/Images/photo-z-scatter-around-mean%s.jpg' % 'w-raw')
        else:
            plt.savefig('C:/Users/iainm/OneDrive/Uni 2018-19/Senior Honours Project/Galaxy-Redshift/Data_analysis/Images/photo-z-scatter-around-mean.jpg')
    plt.show()

def plot_sigma_spec(zipped_sigma, binned_sigma, lin_space_z_spec, i, err=0, plot_raw=True, save=True):
    col=(0,0,1)
    if plot_raw:
        plt.plot(zipped_sigma[:,1], zipped_sigma[:,0], '*', label='Raw data')
        col=(1,0,0)
    plt.plot(lin_space_z_spec[1:-2], binned_sigma[1:-1], color=col, linewidth=2, label='Binned average data')
    # plt.errorbar(lin_space_z_spec[1:-2], binned_sigma[1:-1], color=col, linewidth=2, yerr=err[1:-1], label='Binned average data')
    if plot_raw:
        plt.legend()
    # plt.title('$%d\sigma$ scatter around the mean spec-z vs spectrascopic redshift' % i)
    plt.xlabel('Photometric redshift')
    plt.ylabel('%d$\sigma$ scatter around mean spec-z' % i)
    if save:
        if plot_raw:
            plt.savefig('C:/Users/iainm/OneDrive/Uni 2018-19/Senior Honours Project/Galaxy-Redshift/Data_analysis/Images/z-spec-scatter-%s.jpg' % 'w-raw')
        else:
            plt.savefig('C:/Users/iainm/OneDrive/Uni 2018-19/Senior Honours Project/Galaxy-Redshift/Data_analysis/Images/z-spec-scatter.jpg')
    plt.show()

def plot_bias_spec(zipped_bias, binned_bias, lin_space_z_spec, err=0, plot_raw=True ,save=True):
    col = (0,0,1)
    if plot_raw:
        plt.plot(zipped_bias[:,1], zipped_bias[:,0], '*', label='Raw data')
        col = (1,0,0)
    plt.plot(lin_space_z_spec[1:-2], binned_bias[1:-1], color=col, linewidth=2, label='Binned average data')
    # plt.errorbar(lin_space_z_spec[1:-2], binned_bias[1:-1], color=col, linewidth=2, yerr=err[1:-1], label='Binned average data')
    if plot_raw:
        plt.legend()
    # plt.title('Bias vs spectrascopic redshift')
    plt.xlabel('Photometric redshift')
    plt.ylabel('Bias')
    if save:
        if plot_raw:
            plt.savefig('C:/Users/iainm/OneDrive/Uni 2018-19/Senior Honours Project/Galaxy-Redshift/Data_analysis/Images/spec-bias%s.jpg' % 'w-raw')
        else:
            plt.savefig('C:/Users/iainm/OneDrive/Uni 2018-19/Senior Honours Project/Galaxy-Redshift/Data_analysis/Images/spec-bias.jpg')
    plt.show()


def main():
    BIN_LENGTH = 28
    SHOW_RAW = False
    SAVE_IMG = False
    z_phot, z_spec = read_lephare_data()

    z_spec, z_phot = remove_erronious_data(z_spec, z_phot, len(z_phot))

    # Plot the density map of z-spec vs z-phot
    heatmap, xedges, yedges = plot_densitymap(z_phot, z_spec, save=SAVE_IMG)

    # # Create x-axis linearly spaced for plots.
    # lin_space_z_spec = linspace_data_series(z_spec, BIN_LENGTH+1, 'z_zpec')
    # lin_space_z_phot = linspace_data_series(z_phot, BIN_LENGTH+1, 'z_phot')
    #
    # # Calculate and plot the bias.
    # b = bias(z_phot, z_spec)
    # zipped_bias = zip_data(b, z_spec)
    # binned_bias, std = bin_data_series(zipped_bias, BIN_LENGTH, 'Bias')
    # plot_bias(zipped_bias, binned_bias, lin_space_z_spec, err=std, plot_raw=SHOW_RAW, save=SAVE_IMG)
    #
    # # Calculate and plot the 1 sigma scatter about z-spec
    # sigma = sigma_z(z_phot, z_spec)
    # zipped_sigma = zip_data(sigma, z_spec)
    # binned_sigma, std_sigma = bin_data_series(zipped_sigma, BIN_LENGTH, 'Sigma')
    # plot_sigma(zipped_sigma, binned_sigma, lin_space_z_spec, 1, err=std_sigma, plot_raw=SHOW_RAW, save=SAVE_IMG)
    #
    # # Calculate and plot the 1 sigma scatter about the mean z-phot
    # sigma2 = sigma_z2(z_phot)
    # zipped_sigma2 = zip_data(sigma2, z_spec)
    # binned_sigma2, std_s2 = bin_data_series(zipped_sigma2, BIN_LENGTH, 'Sigma2')
    # plot_sigma_mean(zipped_sigma2, binned_sigma2, lin_space_z_spec, 1, err=std_s2, plot_raw=SHOW_RAW, save=SAVE_IMG)
    #
    # # Calculate and plot the 1 sigma scatter about the mean z-spec
    # sigma3 = sigma_z3(z_spec)
    # zipped_sigma3 = zip_data(sigma3, z_phot)
    # binned_sigma3, std_s3 = bin_data_series(zipped_sigma3, BIN_LENGTH, 'Sigma3')
    # plot_sigma_spec(zipped_sigma3, binned_sigma3, lin_space_z_spec, 1, err=std_s3, plot_raw=SHOW_RAW, save=SAVE_IMG)
    #
    # # Calculate and plot the bias for z-spec.
    # b_spec = bias(z_spec, z_phot)
    # zipped_bias_spec = zip_data(b_spec, z_phot)
    # binned_bias_spec, std_spec = bin_data_series(zipped_bias_spec, BIN_LENGTH, 'Bias spec')
    # plot_bias_spec(zipped_bias_spec, binned_bias_spec, lin_space_z_spec, err=std_spec, plot_raw=SHOW_RAW, save=SAVE_IMG)


    # f_stats(z_phot, z_spec)
    print("*************************************************")


main()
