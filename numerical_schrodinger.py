__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
from threading import Thread
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation




class PartInABox:
    def __init__(self, xs, dx, dt, sigmax, hbar=1, m=1, k0=20, L=20):
        #   Constants
        self.hbar = hbar
        self.m = m
        self.k0 = k0
        self.L = L
        self.omega = self.k0**2*self.hbar / (2*self.m)
        self.norm_const = -1
        self.vg = self.k0*self.hbar / self.m
        self.E = self.hbar * self.omega

        ##  Stepsizes and array sizes
        self.dt = dt
        self.dx = dx
        self.Nx = int(L / self.dx + 1)

        ##  Constants used in Eq. (7a) and (7b).
        self.c1 = self.hbar * self.dt / (2*self.m*self.dx**2)
        self.c2 = self.dt / self.hbar

        #   Initial values/arrays
        self.sigmax = sigmax
        self.V = np.zeros(self.Nx)

        #   Arrays which will update accordingly
        self.x = np.linspace(0, 20, self.Nx)    #   Constant

        self.imArrEven = np.zeros(self.Nx)
        self.reArrEven = np.zeros(self.Nx)
        self.rhoArr = np.zeros(self.Nx)
        self.psiArr = np.zeros(self.Nx, dtype=np.complex128)

        self.imArrOdd = np.zeros(self.Nx)
        self.reArrOdd = np.zeros(self.Nx)
        self.tempArr = np.zeros(self.Nx)

        #   Values which will update accordingly
        self.xs = xs
        self.t = 0
        self.hasPotential = False

        self.normalize()
        self.calc_initial_values()

    def normalize(self):
        not_normalized = np.exp( -(self.x - self.xs)**2 / ( 2 * self.sigmax**2 ) ) * np.exp( 1j * self.k0 * self.x )
        re = np.real(not_normalized)
        im = np.imag(not_normalized)
        abs_squared = re**2 + im**2
        abs_squared_integrated = np.trapz(abs_squared, self.x)  #   Integrating |Psi|^2 using trapes method.

        self.norm_const = 1 / np.sqrt(abs_squared_integrated)
        self.reArrEven = self.norm_const * re
        self.imArrEven = self.norm_const * im
        self.rhoArr = self.norm_const**2 * abs_squared
        self.psiArr = self.reArrEven + 1j * self.imArrEven

    def wave_func(self):
        self.psiArr = self.norm_const * np.exp(-(self.x - self.xs) ** 2 / (2 * self.sigmax ** 2)) * np.exp(1j * ( self.k0 * self.x - self.omega * self.t))

    def calc_initial_values(self):
        self.imArrEven[0], self.imArrEven[self.Nx - 1] = 0, 0
        self.reArrEven[0], self.reArrEven[self.Nx - 1] = 0, 0

        self.t += self.dt/2
        self.wave_func()
        self.imArrOdd = np.imag(self.psiArr)
        self.reArrOdd = np.real(self.psiArr)
        self.imArrOdd[0], self.imArrOdd[self.Nx - 1] = 0, 0
        self.reArrOdd[0], self.reArrOdd[self.Nx - 1] = 0, 0


    def calc_next(self):
        """ANALYTIC
        self.t += self.dt
        self.xs += self.vg*self.dt
        self.wave_func()
        self.reArrEven = np.real(self.psiArr)
        self.imArrEven = np.imag(self.psiArr)
        self.rhoArr = self.imArrEven ** 2 + self.reArrEven ** 2
        """

        #   EVEN
        self.t += self.dt / 2
        self.calc_imArrEven()
        self.calc_reArrEven()

        #   ODD
        self.t += self.dt / 2
        self.calc_imArrOdd()
        self.calc_reArrOdd()

        self.rhoArr = self.reArrEven**2 + self.imArrEven**2
        self.psiArr = self.reArrEven + 1j * self.imArrEven

    def calc_imArrEven(self):
        self.tempArr[1:self.Nx-1] = self.imArrEven[1:self.Nx-1] + self.c1 * (self.reArrOdd[2:self.Nx]
                                    - 2 * self.reArrOdd[1:self.Nx-1] +  self.reArrOdd[0:self.Nx-2])
        self.tempArr -= self.c2 * self.V * self.reArrOdd
        self.imArrEven = np.copy(self.tempArr)

    def calc_reArrEven(self):
        self.tempArr[1:self.Nx-1] = self.reArrEven[1:self.Nx-1] - self.c1 * (self.imArrOdd[2:self.Nx]
                                    - 2 * self.imArrOdd[1:self.Nx-1] + self.imArrOdd[0:self.Nx-2])
        self.tempArr += self.c2 * self.V * self.imArrOdd
        self.reArrEven = np.copy(self.tempArr)

    def calc_imArrOdd(self):
        self.tempArr[1:self.Nx-1] = self.imArrOdd[1:self.Nx - 1] + self.c1 * (self.reArrEven[2:self.Nx]
                                    - 2 * self.reArrEven[1:self.Nx - 1] + self.reArrEven[0:self.Nx - 2])
        self.tempArr -= self.c2 * self.V * self.reArrEven
        self.imArrOdd = np.copy(self.tempArr)

    def calc_reArrOdd(self):
        self.tempArr[1:self.Nx-1] = self.reArrOdd[1:self.Nx - 1] - self.c1 * (self.imArrEven[2:self.Nx]
                                    - 2 * self.imArrEven[1:self.Nx - 1] + self.imArrEven[0:self.Nx - 2])
        self.tempArr += self.c2 * self.V * self.imArrEven
        self.reArrOdd = np.copy(self.tempArr)

    def set_barrier(self, width=None, height=None, V=None):
        #   Creates a barrier with a given potential array V
        if (width == None and height == None):
            self.V = V
            self.hasPotential = not np.equal(self.V,0).all()

        #   Creates a barrier at L/2 with a width and height in factors of E.
        elif (V == None):
            n_half = int(width/self.dx/2)
            n_midpoint = int(self.Nx/2)
            self.V[n_midpoint-n_half:n_midpoint+n_half] = height * self.E
            self.hasPotential = (height != 0)

    def reset(self):
        self.t = 0
        self.wave_func()
        self.imArrEven = np.imag(self.psiArr)
        self.reArrEven = np.real(self.psiArr)
        self.imArrEven[0], self.imArrEven[self.Nx - 1] = 0, 0
        self.reArrEven[0], self.reArrEven[self.Nx - 1] = 0, 0

        self.t += self.dt / 2
        self.wave_func()
        self.imArrOdd = np.imag(self.psiArr)
        self.reArrOdd = np.real(self.psiArr)
        self.imArrOdd[0], self.imArrOdd[self.Nx - 1] = 0, 0
        self.reArrOdd[0], self.reArrOdd[self.Nx - 1] = 0, 0

    def jump_to_time(self, time):
        if (self.t > time):
            self.reset()
        while (self.t < time):
            self.calc_next()

class PartFree(PartInABox):
    def __init__(self, xs, dx, dt, sigmax, hbar=1, m=1, k0=20, L=20):
        super().__init__(xs, dx, dt, sigmax, hbar, m, k0, L)

    def calc_initial_values(self):
        self.t += self.dt/2
        self.wave_func()
        self.imArrOdd = np.imag(self.psiArr)
        self.reArrOdd = np.real(self.psiArr)

    def calc_next(self):
        #   EVEN
        self.t += self.dt / 2
        self.calc_imArrEven()
        self.calc_reArrEven()

        #   ODD
        self.t += self.dt / 2
        self.calc_imArrOdd()
        self.calc_reArrOdd()

        self.rhoArr = self.reArrEven**2 + self.imArrEven**2
        self.psiArr = self.reArrEven + 1j * self.imArrEven

    def calc_imArrEven(self):
        self.tempArr[1:self.Nx-1] = self.imArrEven[1:self.Nx-1] + self.c1 * (self.reArrOdd[2:self.Nx]
                                    - 2 * self.reArrOdd[1:self.Nx-1] +  self.reArrOdd[0:self.Nx-2])
        self.tempArr[self.Nx-1] = self.imArrEven[self.Nx-1] + self.c1 * (self.reArrOdd[0]
                                    - 2 * self.reArrOdd[self.Nx-1] +  self.reArrOdd[self.Nx-2])
        self.tempArr[0] = self.imArrEven[0] + self.c1 * (self.reArrOdd[1]
                                    - 2 * self.reArrOdd[0] + self.reArrOdd[self.Nx-1])
        self.tempArr -= self.c2 * self.V * self.reArrOdd
        self.imArrEven = np.copy(self.tempArr)

    def calc_reArrEven(self):
        self.tempArr[1:self.Nx-1] = self.reArrEven[1:self.Nx-1] - self.c1 * (self.imArrOdd[2:self.Nx]
                                    - 2 * self.imArrOdd[1:self.Nx-1] + self.imArrOdd[0:self.Nx-2])
        self.tempArr[self.Nx - 1] = self.reArrEven[self.Nx - 1] - self.c1 * (self.imArrOdd[0]
                                    - 2 * self.imArrOdd[self.Nx - 1] + self.imArrOdd[self.Nx - 2])
        self.tempArr[0] = self.reArrEven[0] - self.c1 * (self.imArrOdd[1]
                                    - 2 * self.imArrOdd[0] + self.imArrOdd[self.Nx - 1])
        self.tempArr += self.c2 * self.V * self.imArrOdd
        self.reArrEven = np.copy(self.tempArr)

    def calc_imArrOdd(self):
        self.tempArr[1:self.Nx-1] = self.imArrOdd[1:self.Nx - 1] + self.c1 * (self.reArrEven[2:self.Nx]
                                    - 2 * self.reArrEven[1:self.Nx - 1] + self.reArrEven[0:self.Nx - 2])
        self.tempArr[self.Nx - 1] = self.imArrOdd[self.Nx - 1] + self.c1 * (self.reArrEven[0]
                                    - 2 * self.reArrEven[self.Nx - 1] + self.reArrEven[self.Nx - 2])
        self.tempArr[0] = self.imArrOdd[0] + self.c1 * (self.reArrEven[1]
                                    - 2 * self.reArrEven[0] + self.reArrEven[self.Nx - 1])
        self.tempArr -= self.c2 * self.V * self.reArrEven
        self.imArrOdd = np.copy(self.tempArr)

    def calc_reArrOdd(self):
        self.tempArr[1:self.Nx-1] = self.reArrOdd[1:self.Nx - 1] - self.c1 * (self.imArrEven[2:self.Nx]
                                    - 2 * self.imArrEven[1:self.Nx - 1] + self.imArrEven[0:self.Nx - 2])
        self.tempArr[self.Nx - 1] = self.reArrOdd[self.Nx - 1] - self.c1 * (self.imArrEven[0]
                                    - 2 * self.imArrEven[self.Nx - 1] + self.imArrEven[self.Nx - 2])
        self.tempArr[0] = self.reArrOdd[0] - self.c1 * (self.imArrEven[1]
                                    - 2 * self.imArrEven[0] + self.imArrEven[self.Nx - 1])
        self.tempArr += self.c2 * self.V * self.imArrEven
        self.reArrOdd = np.copy(self.tempArr)

    def reset(self):
        self.t = 0
        self.V = np.zeros(self.Nx)
        self.wave_func()
        self.imArrEven = np.imag(self.psiArr)
        self.reArrEven = np.real(self.psiArr)

        self.t += self.dt / 2
        self.wave_func()
        self.imArrOdd = np.imag(self.psiArr)
        self.reArrOdd = np.real(self.psiArr)



class Animation(Thread):
    def __init__(self, part, duration_irl, duration, fps):
        Thread.__init__(self)
        self.part = part
        self.duration_irl = duration_irl
        self.duration = duration
        self.fps = fps
        self.cut_n = int(duration/part.dt/duration_irl/fps)

        # Set up the figure and axes
        self.fig, self.ax1 = plt.subplots()

        # Initialize the line object
        self.line1, = self.ax1.plot([], [], lw=1.0, color='g', label=r"$\Psi_R$")
        self.line2, = self.ax1.plot([], [], lw=1.0, color='m', label=r"$\Psi_I$")
        self.line3, = self.ax1.plot([], [], lw=1.0, color='b', label=r"$|\Psi|^2$")
        self.line4, = self.ax1.plot([], [], lw=1.0, color='k', label=r"$V(x)$")
        self.lines = [self.line1, self.line2, self.line3]
        #   Graphs potential
        if self.part.hasPotential:
            self.line4.set_data(self.part.x, self.part.V/(1.5*self.part.E))
            self.lines.append(self.line4)

        # Set limits and labels for the axes
        self.ax1.set_xlim(left=0, right= part.L)
        self.ax1.set_ylim(bottom=-1, top=1)
        self.ax1.grid()

        # Actually do the animation
        self.anim = animation.FuncAnimation(self.fig, self.animate, repeat=False, frames=int(self.fps * self.duration_irl),
                                interval=1000 / self.fps, blit=False)
        self.filename = "qm_particle_xs=" + str(self.part.xs) + "_sigma=" + str(self.part.sigmax) + "_pot=" + str(self.part.hasPotential) + ".mp4"

    def animate(self, i):
        print(i, "out of", self.fps * self.duration_irl)
        # Math that gets recalculated each iteration
        if (i != 0):
            for j in range(self.cut_n):
                self.part.calc_next()

        # Assigning the line object a set of values
        self.lines[0].set_data(self.part.x, self.part.reArrEven)
        self.lines[1].set_data(self.part.x, self.part.imArrEven)
        self.lines[2].set_data(self.part.x, self.part.rhoArr)

        # Uncomment the following line to save a hi-res version of each frame (mind the filenames though, they'll overwrite each other)
        # plt.savefig('test.png',format='png',dpi=600)

        return self.lines

    def run(self):
        self.anim.save(self.filename, fps=self.fps, extra_args=['-vcodec', 'libx264'], dpi=200, bitrate=-1)

    def run_no_threading(self):
        self.anim.save(self.filename, fps=self.fps, extra_args=['-vcodec', 'libx264'], dpi=200, bitrate=-1)

class Snapshot:
    def __init__(self, part):
        self.part = part
        self.filename = "qm_particle_xs=" + str(self.part.xs) + "_t=" + str(self.part.t-self.part.dt/2) + "_sigma=" + str(self.part.sigmax) + "_pot=" + str(self.part.hasPotential)

    def show_prob_density(self):
        plt.figure()
        plt.plot(self.part.x, self.part.rhoArr, label=r"Prob density, $\rho$", color="b")
        if self.part.hasPotential:
            plt.plot(self.part.x, self.part.V / (1.5 * self.part.E), label=r"Potential", color="k")
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$\rho=|\Psi|^2$", fontsize=16)
        plt.xlim(left=0, right = self.part.L)
        plt.ylim(bottom=0, top = 1)
        plt.legend()
        plt.grid()
        plt.savefig(self.filename + "_probdens.pdf")
        plt.show()

    def show_wave_func(self):
        plt.figure()
        plt.plot(self.part.x, self.part.reArrEven, label=r"$\Psi_R$", color="g")
        plt.plot(self.part.x, self.part.imArrEven, label=r"$\Psi_I$", color="m")
        if self.part.hasPotential:
            plt.plot(self.part.x, self.part.V / (1.5 * self.part.E), label=r"Potential", color="k")
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$\Psi_R$ / $\Psi_I$", fontsize=16)
        plt.xlim(left=0, right=self.part.L)
        plt.ylim(bottom=-1, top=1)
        plt.legend()
        plt.grid()
        plt.savefig(self.filename + "_wavefunc.pdf")
        plt.show()

    def update(self):
        self.filename = "qm_particle_xs=" + str(self.part.xs) + "_t=" + str(self.part.t-self.part.dt/2) + "_sigma=" + str(self.part.sigmax) + "_pot=" + str(self.part.hasPotential)
