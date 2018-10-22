__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan

from numerical_schrodinger import PartInABox, Snapshot, Animation
import numpy as np
from matplotlib import pyplot as plt

def quadratic(x):
    return 0.015*(x-10)**2

if (__name__ == "__main__"):
    #   1 - Save animation, 0 - Do not save
    save_animation = 1

    part1 = PartInABox(10, 0.01, 0.00001, 1) # Format: Start pos xs, stepsize dx, stepsize dt, sigmax
    pot1 = part1.E * quadratic(part1.x)
    part1.set_barrier(V=pot1)

    snap1 = Snapshot(part1)
    snap1.filename = "quadratic_pot"

    ###         ANIMATION       ###
    anim1 = Animation(part1, 20, 2, 60)  # Format: particle object, animation duration, real-life timeframe, fps
    anim1.filename = "quadratic_pot.mp4"
    anims = [anim1]

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