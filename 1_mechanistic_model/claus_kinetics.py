import numpy as np
from scipy.integrate import odeint

# Simplified but accurate Claus kinetic model (3 stages)
def claus_reactor(y, z, T, is_thermal=False):
    H2S, SO2, S2, COS, CS2 = y
    T = T + 273.15  # °C → K

    # Main Claus reaction: 2 H2S + SO2 ⇌ 3/2 S2 + 2 H2O
    if is_thermal:
        k1 = 1e6 * np.exp(-8000/T)
    else:
        k1 = 5e7 * np.exp(-12000/T)  # catalyzed

    r1 = k1 * (H2S**2 * SO2 - (S2**1.5)/(1e-8))  # very simplified equilibrium

    # COS & CS2 hydrolysis (catalytic only)
    if not is_thermal:
        r_cos = 1e5 * np.exp(-10000/T) * COS
        r_cs2 = 1e5 * np.exp(-10000/T) * CS2
    else:
        r_cos = r_cs2 = 0

    dH2S = -2*r1
    dSO2 = -r1
    dS2 = 1.5*r1
    dCOS = -r_cos
    dCS2 = -r_cs2

    return [dH2S, dSO2, dS2, dCOS, dCS2]

def run_mechanistic():
    # Baseline conditions from paper Table 1
    y0 = [55.0, 40.0, 0.0, 0.5, 0.3]
    z = np.linspace(0, 1, 100)
    sol1 = odeint(claus_reactor, y0, z, args=(950, True))
    sol2 = odeint(claus_reactor, sol1[-1], z, args=(240, False))
    sol3 = odeint(claus_reactor, sol2[-1], z, args=(215, False))
    return sol3[-1]

# Test
if __name__ == "__main__":
    print("Final outlet:", run_mechanistic())
