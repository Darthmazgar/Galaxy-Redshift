import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f as f_stats


def read_data(data='Data/zphot.out'):  # zphot.out, 2SLAQ_Original.txt
    return np.genfromtxt(data)


def bias(z_phot, z_spec):
    # return np.average(z_phot - z_spec)
    return z_phot - z_spec


def sigma_z(z_phot, z_spec):
    # sigma = np.sqrt(np.average((z_phot - z_spec)**2))
    sigma = np.sqrt((z_phot - z_spec)**2)
    return sigma


def sigma_z2(z_phot):
    mean = np.average(z_phot)
    # return np.sqrt(np.average((z_phot - mean)**2))
    return np.sqrt((z_phot - mean)**2)


def sigma_z3(z_spec):
    mean = np.average(z_spec)
    # return np.sqrt(np.average((z_spec - mean)**2))
    return np.sqrt((z_spec - mean)**2)


def plot_sig_photoz(sigma, z_spec, n_data_sets=1):
    # sigma = [sigma for x in range(len(z_phot))]
    plt.plot(z_spec, sigma, '*')
    plt.ylabel("$1\sigma$ scatter around mean spec-z")
    plt.xlabel("Spectroscopic Redshift")
    plt.show()


def plot_bias(bias, z_spec, n_data_sets=1):
    # z_spec = [z_spec[i] if z_spec[i] != 0.999900E+01 else 0 for i in range(len(z_spec))]  # It is odd how much data == 0.999900E+01
    plt.plot(z_spec, bias, '*')
    plt.xlabel("Spectroscopic Redshift")
    plt.ylabel("Bias")
    plt.show()


def main():
    data = read_data()
    # z_phot = np.log10(data[:, 1])  # for zphot.out
    z_phot = data[:, 1]
    # print(z_phot)
    # z_spec = np.log10(data[:, 22])
    z_spec = data[:, 22]

    # print(z_spec)

    sigma = sigma_z(z_phot, z_spec)
    # print(sigma)
    plot_sig_photoz(sigma, z_phot)

    b = bias(z_phot, z_spec)
    plot_bias(b, z_spec)

main()
