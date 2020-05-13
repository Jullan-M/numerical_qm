__author__ = 'Jullan'
# -*- coding: utf-8 -*-
# Made by Jullan
from threading import Thread
import numpy as np
from scipy.integrate import simps
from matplotlib import pyplot as plt
from matplotlib import animation
import mplcyberpunk

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{eufrak}')

HBAR = 1


class PartInABox:
    def __init__(self, xs, dx, dt, sigmax, k0=20, m=1, L=20):
        #   Constants
        self.k0 = k0
        self.L = L

        self.omega = self.k0**2*HBAR / (2*m)
        self.E = HBAR * self.omega

        self.xs = xs
        self.sigmax = sigmax

        # Stepsizes and array sizes
        self.dt = dt
        self.dx = dx
        self.Nx = int(self.L / self.dx + 1)

        # Constants used in Eq. (7a) and (7b).
        self.c1 = HBAR * self.dt / (2*m*self.dx**2)
        self.c2 = self.dt / HBAR

        #   Arrays which will update accordingly
        self.x = np.linspace(0, self.L, self.Nx)  # Constant
        self.V = np.zeros(self.Nx)

        self.imArrEven = np.zeros(self.Nx)
        self.reArrEven = np.zeros(self.Nx)
        self.rhoArr = np.zeros(self.Nx)
        self.psiArr = np.zeros(self.Nx, dtype=np.complex128)

        self.imArrOdd = np.zeros(self.Nx)
        self.reArrOdd = np.zeros(self.Nx)

        #   Values which will update accordingly
        self.t = 0
        self.hasPotential = False

        #   These will normalize and calculate the initial values
        #   for psi during the initialization of particle object.
        self.normalize()
        self.calc_initial_values()

    def normalize(self):
        #   Normalizes the wave-function.
        not_normalized = np.exp(-(self.x - self.xs)**2 /
                                (2 * self.sigmax**2)) * np.exp(1j * self.k0 * self.x)
        re = np.real(not_normalized)
        im = np.imag(not_normalized)
        abs_squared = re**2 + im**2
        #   Integrating |Psi|^2 using simpsons method
        abs_squared_integrated = simps(abs_squared, self.x)

        self.norm_const = 1 / np.sqrt(abs_squared_integrated)
        self.reArrEven = self.norm_const * re
        self.imArrEven = self.norm_const * im
        self.rhoArr = self.norm_const**2 * abs_squared
        self.psiArr = self.reArrEven + 1j * self.imArrEven

    def wave_func(self):
        #   Eq. (8) in the exercise description.
        self.psiArr = self.norm_const * np.exp(-(self.x - self.xs) ** 2 / (
            2 * self.sigmax ** 2)) * np.exp(1j * (self.k0 * self.x - self.omega * self.t))

    def calc_initial_values(self):
        #   Forcing the edges of the wave-function to be zero.
        #   Effectively making this an particle in a box.
        self.imArrEven[0], self.imArrEven[-1] = 0, 0
        self.reArrEven[0], self.reArrEven[-1] = 0, 0

        self.t += self.dt/2
        self.wave_func()
        self.imArrOdd = np.imag(self.psiArr)
        self.reArrOdd = np.real(self.psiArr)
        self.imArrOdd[0], self.imArrOdd[-1] = 0, 0
        self.reArrOdd[0], self.reArrOdd[-1] = 0, 0

    def calc_next(self):
        #   EVEN (t = 0, dt, 2dt, 3dt, ...)
        self.t += self.dt / 2
        self.calc_imArr(self.imArrEven, self.reArrOdd)
        self.calc_reArr(self.reArrEven, self.imArrOdd)

        #   ODD (t = 1/2 dt, 3/2 dt, 5/2 dt, ...)
        self.t += self.dt / 2
        self.calc_imArr(self.imArrOdd, self.reArrEven)
        self.calc_reArr(self.reArrOdd, self.imArrEven)

        self.rhoArr = self.reArrEven**2 + self.imArrEven**2
        self.psiArr = self.reArrEven + 1j * self.imArrEven

    def calc_imArr(self, imArr, reArr):
        imArr[1:-1] = imArr[1:-1] - self.c2 * self.V[1:-1] * reArr[1:-1] + \
            self.c1 * (reArr[2:] - 2 * reArr[1:-1] + reArr[:-2])

    def calc_reArr(self, reArr, imArr):
        reArr[1:-1] = reArr[1:-1] + self.c2 * self.V[1:-1] * imArr[1:-1] - \
            self.c1 * (imArr[2:] - 2 * imArr[1:-1] + imArr[:-2])

    def set_barrier(self, width=None, height=None, V=None):
        #   Creates a potential field with a given potential array V.
        if (width == None and height == None):
            self.V = V
            self.hasPotential = not np.equal(self.V, 0).all()

        #   Creates a barrier at L/2 with a width and height in factors of E.
        elif (V == None):
            n_half = int(width/self.dx/2)
            n_midpoint = int(self.Nx/2)
            self.V[n_midpoint-n_half:n_midpoint+n_half] = height * self.E
            self.hasPotential = (height != 0)

    def reset(self):
        #   This resets the whole object back to its initial conditions.
        #   The potential V remains the same.
        self.t = 0
        self.wave_func()
        self.imArrEven = np.imag(self.psiArr)
        self.reArrEven = np.real(self.psiArr)
        self.imArrEven[0], self.imArrEven[-1] = 0, 0
        self.reArrEven[0], self.reArrEven[-1] = 0, 0

        self.t += self.dt / 2
        self.wave_func()
        self.imArrOdd = np.imag(self.psiArr)
        self.reArrOdd = np.real(self.psiArr)
        self.imArrOdd[0], self.imArrOdd[-1] = 0, 0
        self.reArrOdd[0], self.reArrOdd[-1] = 0, 0

    def jump_to_time(self, time):
        #   Function that jumps to a specific time.
        #   Useful for snapshots, but should be used after rendering an animation.
        if (self.t > time):
            self.reset()
        while (self.t < time):
            self.calc_next()


class PartFree(PartInABox):
    def __init__(self, xs, dx, dt, sigmax, k0=20, m=1, L=20):
        super().__init__(xs, dx, dt, sigmax, k0, m, L)

    def calc_initial_values(self):
        self.t += self.dt/2
        self.wave_func()
        self.imArrOdd = np.imag(self.psiArr)
        self.reArrOdd = np.real(self.psiArr)

    def calc_imArr(self, imArr, reArr):
        imArr[1:-1] = imArr[1:-1] - self.c2 * self.V[1:-1] * reArr[1:-1] + \
            self.c1 * (reArr[2:] - 2 * reArr[1:-1] + reArr[:-2])
        imArr[-1] = imArr[-1] - self.c2 * self.V[-1] * reArr[-1] + \
            self.c1 * (reArr[0] - 2 * reArr[-1] + reArr[-2])
        imArr[0] = imArr[0] - self.c2 * self.V[0] * reArr[0] + \
            self.c1 * (reArr[1] - 2 * reArr[0] + reArr[-1])

    def calc_reArr(self, reArr, imArr):
        reArr[1:-1] = reArr[1:-1] + self.c2 * self.V[1:-1] * imArr[1:-1] - \
            self.c1 * (imArr[2:] - 2 * imArr[1:-1] + imArr[:-2])
        reArr[-1] = reArr[-1] + self.c2 * self.V[-1] * imArr[-1] - \
            self.c1 * (imArr[0] - 2 * imArr[-1] + imArr[-2])
        reArr[0] = reArr[0] + self.c2 * self.V[0] * imArr[0] - \
            self.c1 * (imArr[1] - 2 * imArr[0] + imArr[-1])

    def reset(self):
        self.t = 0
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

        plt.style.use("cyberpunk")
        # Set up the figure and axes
        self.fig, self.ax1 = plt.subplots()
        self.fig.set_size_inches(16, 9, True)
        self.fig.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.ax1.set_axis_off()
        self.ax1.text(
            17.5, -0.9, r"$@ \mathfrak{J}\textrm{ullan\_M}$", fontsize=20)

        # Initialize the line object
        self.line1, = self.ax1.plot(
            [], [], lw=0.75, color='c', label=r"$\Psi_R$")
        self.line2, = self.ax1.plot(
            [], [], lw=0.75, color='m', label=r"$\Psi_I$")
        self.line3, = self.ax1.plot(
            [], [], lw=0.75, color='y', label=r"$|\Psi|^2$")
        self.line4, = self.ax1.plot([], [], lw=1.0, color='w', label=r"$V(x)$")
        self.lines = [self.line1, self.line2, self.line3]
        #   Plots potential (NOT in the same units at all)
        if self.part.hasPotential:
            self.line4.set_data(self.part.x, self.part.V/(3*self.part.E))
            self.lines.append(self.line4)

        self.nl = len(self.lines)  # number of lines (not including glow lines)

        for l in range(len(self.lines)):
            for n in range(1, 11):
                self.lines.append(self.ax1.plot([], [])[0])

        # Set limits and labels for the axes
        self.ax1.set_xlim(left=0, right=part.L)
        self.ax1.set_ylim(bottom=-1, top=1)

        # Actually do the animation
        self.anim = animation.FuncAnimation(self.fig, self.animate, repeat=False, frames=int(self.fps * self.duration_irl),
                                            interval=1000 / self.fps, blit=False)
        self.filename = "qm_particle_xs=" + str(self.part.xs) + "_sigma=" + str(
            self.part.sigmax) + "_pot=" + str(self.part.hasPotential) + ".webm"

    def animate(self, i):
        # Math that gets recalculated each iteration
        if (i != 0):
            for _ in range(self.cut_n):
                self.part.calc_next()

        # Assigning the line object a set of values
        self.lines[0].set_data(self.part.x, self.part.reArrEven)
        self.lines[1].set_data(self.part.x, self.part.imArrEven)
        self.lines[2].set_data(self.part.x, self.part.rhoArr)

        for l in range(self.nl):
            data = self.lines[l].get_data()
            linewidth = self.lines[l].get_linewidth()
            for n in range(0, 10):
                index = self.nl + n + l * 10
                self.lines[index].set_data(data)
                # line properties are copied as seen in this solution: https://stackoverflow.com/a/54688412/3240855
                self.lines[index].update_from(self.lines[l])

                self.lines[index].set_alpha(0.03)
                self.lines[index].set_linewidth(
                    linewidth + 1.05 * (n+1))
                # mark the glow lines, to disregard them in the underglow function.
                self.lines[index].is_glow_line = True

        # Uncomment the following line to save a hi-res version of each frame (mind the filenames though, they'll overwrite each other)
        # plt.savefig('test.png',format='png',dpi=600)

        return self.lines

    def run(self):
        self.anim.save(self.filename, fps=self.fps, extra_args=[
                       '-vcodec', 'libvpx-vp9'], dpi=120, bitrate=4000, progress_callback=lambda i, n: print(f'{i} out of {n}.'))

    def run_no_threading(self):
        self.anim.save(self.filename, fps=self.fps, extra_args=[
                       '-vcodec', 'libvpx-vp9'], dpi=120, bitrate=4000, progress_callback=lambda i, n: print(f'{i} out of {n}.'))


class Snapshot:
    def __init__(self, part):
        self.part = part
        self.filename = "qm_particle_xs=" + str(self.part.xs) + "_t=" + str(
            self.part.t-self.part.dt/2) + "_sigma=" + str(self.part.sigmax) + "_pot=" + str(self.part.hasPotential)
        self.save_fig = True

    def show_prob_density(self):
        plt.figure()
        plt.plot(self.part.x, self.part.rhoArr,
                 label=r"Prob density, $\rho$", color="b")
        if self.part.hasPotential:
            plt.plot(self.part.x, self.part.V / (1.5 * self.part.E),
                     label=r"Potential", color="k")
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$\rho=|\Psi|^2$", fontsize=16)
        plt.xlim(left=0, right=self.part.L)
        plt.ylim(bottom=0, top=1)
        plt.legend()
        plt.grid()
        if self.save_fig:
            plt.savefig(self.filename + "_probdens.pdf")
        plt.show()

    def show_wave_func(self):
        plt.figure()
        plt.plot(self.part.x, self.part.reArrEven,
                 label=r"$\Psi_R$", color="g", linewidth=0.75)
        plt.plot(self.part.x, self.part.imArrEven,
                 label=r"$\Psi_I$", color="m", linewidth=0.75)
        if self.part.hasPotential:
            plt.plot(self.part.x, self.part.V / (1.5 * self.part.E),
                     label=r"Potential", color="k")
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$\Psi_R$ / $\Psi_I$", fontsize=16)
        plt.xlim(left=0, right=self.part.L)
        plt.ylim(bottom=-1, top=1)
        plt.legend()
        plt.grid()
        if self.save_fig:
            plt.savefig(self.filename + "_wavefunc.pdf")
        plt.show()

    def show_both(self):
        plt.figure()
        plt.plot(self.part.x, self.part.reArrEven,
                 label=r"$\Psi_R$", color="g", linewidth=0.75)
        plt.plot(self.part.x, self.part.imArrEven,
                 label=r"$\Psi_I$", color="m", linewidth=0.75)
        plt.plot(self.part.x, self.part.rhoArr,
                 label=r"Prob density, $\rho$", color="b")
        if self.part.hasPotential:
            plt.plot(self.part.x, self.part.V / (1.5 * self.part.E),
                     label=r"Potential", color="k")
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$\Psi_R$ / $\Psi_I$", fontsize=16)
        plt.xlim(left=0, right=self.part.L)
        plt.ylim(bottom=-1, top=1)
        plt.legend()
        plt.grid()
        if self.save_fig:
            plt.savefig(self.filename + "_both.pdf")
        plt.show()

    def update(self):
        self.filename = "qm_particle_xs=" + str(self.part.xs) + "_t=" + str(
            self.part.t-self.part.dt/2) + "_sigma=" + str(self.part.sigmax) + "_pot=" + str(self.part.hasPotential)


if (__name__ == "__main__"):
    part1 = PartInABox(5, 0.01, 0.00001, 1.5)
    part1.set_barrier(part1.L/50, 1.1)
    part1.jump_to_time(0.5)
    plt.figure()
    plt.plot(part1.x, part1.reArrEven, color="g", linewidth=0.75)
    plt.plot(part1.x, part1.imArrEven, color="m", linewidth=0.75)
    plt.xlim(left=0, right=part1.L)
    plt.axis("off")
    plt.savefig("front_page.pdf", bbox_inches='tight')
    plt.show()
