__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
from numerical_schrodinger import PartFree, Animation, Snapshot
from matplotlib import pyplot as plt


if (__name__ == "__main__"):
    #   1 - Save animation, 0 - Do not save
    save_animation = 0

    free1 = PartFree(15, 0.01, 0.00001, 1) # Format: Start pos xs, stepsize dx, stepsize dt, sigmax

    ###         ANIMATION       ###
    anim1 = Animation(free1, 10, 0.5, 60)  # Format: particle object, animation duration, real-life timeframe, fps
    anim1.filename = "free_particle.mp4"
    anims = [anim1]

    if save_animation == 1:
        for anim in anims:
            anim.run_no_threading()

    plt.show()

    ###         PLOTS           ###
    free1.jump_to_time(0.175)
    snap1 = Snapshot(free1)
    snap1.show_wave_func()
    free1.jump_to_time(0.25)
    snap1.update()
    snap1.show_wave_func()
    free1.jump_to_time(0.325)
    snap1.update()
    snap1.show_wave_func()

