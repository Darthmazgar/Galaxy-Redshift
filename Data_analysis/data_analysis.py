import numpy as np
import matplotlib.pyplot as plt


def read_data(data='Data/zphot.out'):  # zphot.out, 2SLAQ_Original.txt
    return np.genfromtxt(data)


def one_sigma_scatter(z_phot, z_spec):
    # sigma = np.sqrt(np.average((z_phot - z_spec)**2))
    sigma = np.sqrt((z_phot - z_spec)**2)
    return sigma


def bias(z_phot, z_spec):
    # return np.average(z_phot - z_spec)
    return z_phot - z_spec




def plot_sig_photoz(sigma, z_phot, n_data_sets=1):
    # sigma = [sigma for x in range(len(z_phot))]
    plt.plot(z_phot, sigma, '*')
    plt.ylabel("$1\sigma$ scatter around mean spec-z")
    plt.xlabel("Photometric Redshift")
    plt.show()


def plot_bias(bias, z_spec, n_data_sets=1):
    plt.plot(z_spec, bias, '*')
    plt.show()

def main():
    data = read_data()
    z_phot = np.log10(data[:, 1])  # for zphot.out
    # print(z_phot)
    z_spec = np.log10(data[:, 22])

    # print(z_spec)

    sigma = one_sigma_scatter(z_phot, z_spec)
    # print(sigma)
    plot_sig_photoz(sigma, z_phot)

    b = bias(z_phot, z_spec)
    plot_bias(b, z_spec)

main()
