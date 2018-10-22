__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
from numerical_schrodinger import PartInABox, Animation, Snapshot
from matplotlib import pyplot as plt


if (__name__ == "__main__"):
    #   1 - Save animation, 0 - Do not save
    save_animation = 1

    part1 = PartInABox(5, 0.01, 0.00001, 0.5) # Format: Start pos xs, stepsize dx, stepsize dt, sigmax
    snap1 = Snapshot(part1)

    part2 = PartInABox(5, 0.01, 0.00001, 1)
    snap2 = Snapshot(part2)

    part3 = PartInABox(5, 0.01, 0.00001, 1.5)
    snap3 = Snapshot(part3)

    part4 = PartInABox(5, 0.01, 0.00001, 2)
    snap4 = Snapshot(part4)

    test = PartInABox(5, 0.01, 0.001, 1)
    snap5 = Snapshot(test)
    snap5.filename = "high_timestep_initial"

    anim1 = Animation(part1, 10, 0.5, 60)   #   Format: particle object, animation duration, real-life timeframe, fps
    anim2 = Animation(part2, 10, 0.5, 60)
    anim3 = Animation(part3, 10, 0.5, 60)
    anim4 = Animation(part4, 10, 0.5, 60)
    anim5 = Animation(test, 10, 0.5, 60)
    anim5.filename = "high_timestep.mp4"

    parts = [part1,
             part2,
             part3,
             part4,
             test]
    anims = [anim1,
             anim2,
             anim3,
             anim4,
             anim5]
    snaps = [snap1,
             snap2,
             snap3,
             snap4,
             snap5]

    for snap in snaps:
        snap.show_prob_density()

    if save_animation == 1:
        for anim in anims:
            anim.run_no_threading()

    for part in parts:
        part.jump_to_time(0.5)

    for i in range(4):
        snaps[i].update()
        snaps[i].show_prob_density()
    snaps[4].filename = "high_timestep_after"
    snaps[4].show_prob_density()

    plt.show()