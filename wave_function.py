__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
from matplotlib import pyplot as plt



class PartInABox:
    def __init__(self, xs, dx, dt, sigmax, hbar=1, m=1, k0=20, L=20):
        #   Constants
        self.hbar = hbar
        self.m = m
        self.k0 = k0
        self.L = L
        self.omega = 2*self.k0**2*self.hbar / self.m
        self.norm_const = -1

        #   Initial values/arrays
        self.xs = xs
        self.sigmax = sigmax
        self.reArrInitial = np.zeros(self.Nx)
        self.imArrInitial = np.zeros(self.Nx)

        #   Stepsizes and array sizes
        self.dx = dx
        self.Nx = int(L / self.dx + 1)

        #   Arrays which will update accordingly
        self.x = np.linspace(0, 20, self.Nx)    #   Constant
        self.reArr = np.zeros(self.Nx)
        self.imArr = np.zeros(self.Nx)
        self.rhoArr = np.zeros(self.Nx)
        self.psiArr = np.zeros(self.Nx, dtype=np.complex128)
        self.E = self.hbar * self.omega * np.ones(self.Nx)

        #   Values which will update accordingly
        tRe = 0
        tIm = -dt/2

        self.normalize()

    def normalize(self):
        not_normalized = np.exp( -(self.x - self.xs)**2 / ( 2 * self.sigmax**2 ) ) * np.exp( 1j * self.k0 * self.x )
        re = np.real(not_normalized)
        im = np.imag(not_normalized)
        abs_squared = re**2 + im**2
        abs_squared_integrated = np.trapz(abs_squared, self.x)  #   Integrating |Psi|^2 using trapes method.

        self.norm_const = 1 / np.sqrt(abs_squared_integrated)
        self.reArr = self.norm_const * re
        self.imArr = self.norm_const * im
        self.rhoArr = self.norm_const**2 * abs_squared
        self.psiArr = self.reArr + 1j * self.imArr

    def calc_initial_values(self):  #This function assumes that normalize() has been ran beforehand.
        self.imArrInitial = self.imArr
        self.reArrInitial = se

    def wave_func(self):
        return self.norm_const * np.exp(-(self.x - self.xs) ** 2 / (2 * self.sigmax ** 2)) * np.exp(1j * ( self.k0 * self.x - self.omega * self.t))




if (__name__ == "__main__"):
    part1 = PartInABox(5, 0.001, 1)
    plt.figure(1)
    plt.plot(part1.x, part1.reArr, label=r"$\Psi_R$", color="g")
    plt.plot(part1.x, part1.imArr, label=r"$\Psi_I$", color="m")
    plt.plot(part1.x, part1.rhoArr, label=r"$\Psi^2$", color="b")
    plt.xlabel(r"$x$", fontsize=16)
    plt.ylabel(r"$\Psi_R$ / $\Psi_I$ / $\Psi^2$", fontsize=16)
    plt.xlim(left=0, right= part1.L)
    plt.legend()
    plt.grid()
    plt.show()