__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
from numerical_schrodinger import PartInABox, Snapshot

if (__name__ == "__main__"):
    part1 = PartInABox(5, 0.01, 0.00001, 1) # Format: Start pos xs, stepsize dx, stepsize dt, sigmax
    snap1 = Snapshot(part1)
    snap1.show_wave_func()
    snap1.show_prob_density()