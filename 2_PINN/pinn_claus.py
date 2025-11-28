import deepxde as dde
import numpy as np

# Domain
geom = dde.geometry.Interval(0, 1)

def pde(z, y):
    H2S, SO2, T = y[:,0:1], y[:,1:2], y[:,2:3]
    dH2S_dz = dde.grad.jacobian(y, 0, j=0)
    dSO2_dz = dde.grad.jacobian(y, 1, j=0)
    dT_dz = dde.grad.jacobian(y, 2, j=0)
    # Simplified physics residual
    return [
        dH2S_dz + 2*(H2S**2 * SO2)*1e-3,
        dSO2_dz + (H2S**2 * SO2)*1e-3,
        dT_dz - 50  # exothermic
    ]

def boundary_l(z, on_boundary):
    return on_boundary

data = dde.data.PDE(geom, pde, [], num_domain=2000, num_boundary=100)
net = dde.maps.FNN([1] + [50]*4 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=1e-4)
model.train(epochs=10000)
