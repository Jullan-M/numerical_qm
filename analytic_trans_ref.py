__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 10/3, 200)
k0 = 20
hbar = 1
a = 2/5
m = 1
E = k0**2*hbar**2 / (2*m)

#
def transmission_h(x):
    if (x==0):
        return x-x
    elif (x < 1):
        return 1/(1+np.sinh(np.sqrt(2 * m * E * (1/x - 1)/hbar**2)*a)**2/(4*x*(1-x)))
    elif (x > 1):
        return 1/(1+np.sin(np.sqrt(2 * m * E * (1 - 1/x) / hbar ** 2)*a)**2/(4*x*(x-1)))
    else:
        return 1/(1 + m*a**2*E/(2*hbar**2))+x-x

w = np.linspace(0, 1, 200)

def transmission_w(w, x = 10/9):
    return 1/(1+np.sin(np.sqrt(2 * m * E * (1 - 1/x) / hbar ** 2)*w)**2/(4*x*(x-1)))

vtransmission_h = np.vectorize(transmission_h)
print(print(1-vtransmission_h(1.182330173)))
plt.figure()
plt.plot(x, vtransmission_h(x), label=r"Transmission prob", color="b")
plt.plot(x, 1-vtransmission_h(x), label=r"Reflection prob", color="r")
plt.xlabel(r"$E/V_0$", fontsize=16)
plt.ylabel(r"Probability", fontsize=16)
plt.xlim(left=2/3,right=10/3)
plt.ylim(bottom=0)
plt.legend()
plt.grid()
plt.savefig("p4_ref_vs_trans_height_analytic.pdf")
plt.show()

plt.figure()
plt.plot(w, transmission_w(w), label=r"Transmission prob", color="b")
plt.plot(w, 1-transmission_w(w), label=r"Reflection prob", color="r")
plt.xlabel(r"Barrier width, $l$", fontsize=16)
plt.ylabel(r"Probability", fontsize=16)
plt.xlim(left=0, right=1)
plt.ylim(bottom=0, top=1)
plt.legend()
plt.grid()
plt.savefig("p5_ref_vs_trans_width_analytic.pdf")
plt.show()