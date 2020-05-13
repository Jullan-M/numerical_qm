__author__ = 'Jullan'
# -*- coding: utf-8 -*-
# Made by Jullan

from numerical_schrodinger import PartInABox, PartFree, Snapshot, Animation
import numpy as np
from matplotlib import pyplot as plt


def quadratic(x):
    return 0.015*(x-10)**2


if (__name__ == "__main__"):
    #   1 - Save animation, 0 - Do not save
    save_animation = 1

    # Format: Start pos xs, stepsize dx, stepsize dt, sigmax
    part1 = PartInABox(10, 0.01, 0.00001, 1)
    part2 = PartInABox(5, 0.01, 0.00001, 1)
    part3 = PartInABox(5, 0.01, 0.00001, 1)

    #   Setting up potentials.
    # Quadratic potential
    pot1 = part1.E * quadratic(part1.x)
    part1.set_barrier(V=pot1)
    # Triangular potential
    pot2 = np.zeros(part1.Nx)
    width = part1.L/50
    n_mid = int(part1.Nx/2)
    n_half = int(width/part1.dx/2)
    pot2[n_mid-n_half:n_mid+n_half] = part1.E * \
        (1 - 0.5 * (part1.x[n_mid-n_half:n_mid+n_half]-10)
         )  # Upper: 1.1; Lower: 0.9
    part2.set_barrier(V=pot2)
    # Regular barrier
    part3.set_barrier(part3.L/50, 0.78)

    snap1 = Snapshot(part1)
    snap1.filename = "quadratic_pot"
    snap2 = Snapshot(part2)
    snap2.filename = "triangular_pot"
    snap3 = Snapshot(part3)
    snap3.filename = "flat_barrier_pot"

    ###         ANIMATION       ###
    # Format: particle object, animation duration, real-life timeframe, fps
    anim1 = Animation(part1, 20, 2, 60)
    anim1.filename = "quadratic_pot.mp4"
    anim2 = Animation(part2, 12, 0.6, 60)
    anim2.filename = "triangular_pot.mp4"
    anim3 = Animation(part3, 10, 0.5, 60)
    anim3.filename = "flat_barrier_pot.mp4"
    anims = [  # anim1,
        # anim2,
        anim3]

    if save_animation == 1:
        for anim in anims:
            anim.run_no_threading()

    plt.show()
    """
    ###         PLOTS           ###
    part1.jump_to_time(0.5)
    snap1.show_wave_func()
    snap1.show_prob_density()
    ref_prob = np.trapz(part1.rhoArr[:int(part1.Nx / 2)], dx=part1.dx)
    trans_prob = 1-ref_prob
    print("Reflection probability =", ref_prob)
    print("Transmission probability =", trans_prob)
    """
