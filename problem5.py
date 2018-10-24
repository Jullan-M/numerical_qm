__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
from numerical_schrodinger import PartInABox
import numpy as np
from matplotlib import pyplot as plt

if (__name__ == "__main__"):
    #   1 - Save animation, 0 - Do not save
    part1 = PartInABox(5, 0.010005002, 0.00004, 1)  # Format: Start pos xs, stepsize dx, stepsize dt, sigmax

    n_bar = 50
    ref_prob_arr = np.zeros(n_bar)
    trans_prob_arr = np.zeros(n_bar)
    for i in range(n_bar):
        print(i, "out of", n_bar)
        part1.reset()
        part1.set_barrier(part1.L / 20 * (i+1)/n_bar, 9/10)
        part1.jump_to_time(0.5)
        ref_prob = np.trapz(part1.rhoArr[:int(part1.Nx / 2)], part1.x[:int(part1.Nx / 2)])
        ref_prob_arr[i] = ref_prob
        trans_prob_arr[i] = 1 - ref_prob

    width = np.linspace(part1.L / (20*n_bar), part1.L/20, n_bar)
    plt.figure()
    plt.plot(width, ref_prob_arr, label=r"Reflection prob", color="r")
    plt.plot(width, trans_prob_arr, label=r"Transmission prob", color="b")
    plt.xlabel(r"Barrier width, $l$", fontsize=16)
    plt.ylabel(r"Probability", fontsize=16)
    plt.xlim(left=0, right=part1.L/20)
    plt.ylim(bottom=0, top=1)
    plt.legend()
    plt.grid()
    plt.savefig("p5_ref_vs_trans_width_graph.pdf")
    plt.show()