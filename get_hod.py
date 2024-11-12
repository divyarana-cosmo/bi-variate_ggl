import numpy as np
from scipy.integrate import simps

class bivariate:
    def __init__(self, alpha, beta, gamma, delta, sig0, sig1, sigx):
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.delta  = delta
        self.sig0   = sig0
        self.sig1   = sig1
        self.sigx   = sigx

    #define the scaling relations
    def avglog_lum(self, alpha, beta, logmh):
        "luminosity-halo mass relation"
        return alpha*(logmh - 12)  + beta)

    def avglog_rich(self, gamma, delta, logmh):
        "richness-halo mass relation"
        return alpha*(logmh - 12)  + beta

# write the lognormal

# derive the hod via integral 8.5,16.5 h-1Msun
