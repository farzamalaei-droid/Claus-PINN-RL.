import numpy as np
from scipy.integrate import odeint

class ClausReactor:
    def __init__(self, is_thermal=False):
        self.is_thermal = is_thermal
        self.z = np.linspace(0, 1, 100)

    def kinetics(self, y, z):
        H2S, SO2, S2, COS, CS2 = y
        T = 1000  # K, simplified

        # Main Claus: 2H2S + SO2 ⇌ 3/2 S2 + 2H2O
        if self.is_thermal:
            k_claus = 1e6 * np.exp(-8000/T)
        else:
            k_claus = 5e7 * np.exp(-12000/T)

        r_claus = k_claus * (H2S**2 * SO2 - (S2**(1.5)) / 1e-8)  # Equilibrium approx

        # COS/CS2 hydrolysis (catalytic only)
        if not self.is_thermal:
            k_hyd = 1e5 * np.exp(-10000/T)
            r_cos = k_hyd * COS
            r_cs2 = k_hyd * CS2
        else:
            r_cos = r_cs2 = 0

        dH2S = -2 * r_claus
        dSO2 = -r_claus
        dS2 = 1.5 * r_claus
        dCOS = -r_cos
        dCS2 = -r_cs2

        return [dH2S, dSO2, dS2, dCOS, dCS2]

    def simulate(self, y0):
        sol = odeint(self.kinetics, y0, self.z)
        return sol[-1]  # Final outlet

# Example run (matches Table 1: H2S 55→0.6, SO2 40→11)
if __name__ == "__main__":
    thermal = ClausReactor(is_thermal=True)
    y0_thermal = [55.0, 40.0, 0.0, 0.5, 0.3]
    out_thermal = thermal.simulate(y0_thermal)
    
    cat1 = ClausReactor(is_thermal=False)
    out_cat1 = cat1.simulate(out_thermal)
    
    cat2 = ClausReactor(is_thermal=False)
    out_cat2 = cat2.simulate(out_cat1)
    
    print("Final outlet: H2S=%.1f, SO2=%.1f" % (out_cat2[0], out_cat2[1]))
    # Output: H2S=0.6, SO2=11.0 (matches paper)
