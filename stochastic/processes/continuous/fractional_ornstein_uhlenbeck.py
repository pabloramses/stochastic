import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import warnings
import scipy as sci
from utils import ensure_single_arg_constant_function
from stochastic.processes.continuous import FractionalBrownianMotion
r"""Fractional Ornstein-Uhlenbeck process.

    A fractional Ornstein-Uhlenbeck of hurst parameter 0<H<1 genearlises the Ornstein-Uhlenbeck 
    path-wise in the same manner as fractional Brownian motion does to a standard Brownian motion,
    introducing a parameter H that controls the Hölder-continuity of the paths. Informaly, 
    paths generated from an fOU process will have an oscillating qualitative behaviour similar to 
    a standard Ornstein-Uhlenbeck process but will be locally more or less rough depending on H. It 
    is formally defined as the solution to 
    .. math::

        dX_t = -\theta_t X_t dt + \sigma_t X_t^{\gamma_t} dW_t^H
  

    where W_t^H is a fractional Brownian motion with Hurst parameter H. 

    Simulations are obtained through a first order Euler-Maruyama approximation, which ensures a strong 
    convergence of at least order H. 
"""

class fOU:
    def __init__(self, H, T, sigma = 1, epsilon = 0.1):
        self.H = H
        self.T = T 
        self.fbm = self.fractional_noise
        self.sigma = sigma 
        self.scale = epsilon 

    @property
    def fractional_noise(self):
        return FractionalBrownianMotion(hurst = self.H, t = self.T)
    
    @property
    def scale(self): 
        return self._scale 
    
    @scale.setter 
    def scale(self, value): 
        self._scale = ensure_single_arg_constant_function(value) 

    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        self._sigma = ensure_single_arg_constant_function(value)

    def noise(self, n): 
        self.noise = self.fractional_noise.sample(n+1)

    def sample(self, n, initial = 0.0):
        """
        Employs a first order Euler-Maruyama scheme with the precaution that the traslation of our notation to conventional is:
        speed = 1/scale
        mean = 0
        volatility = sigma/(epsilon^H)
        exponential volatility = 0
        """
        if self.H == 0.5: 
            initial = np.random.normal(0,self._sigma(0)/np.sqrt(2))
        else: 
            variance = ((self._sigma(0)**2)*(self._scale(0)**(2*self.H))/2)*sci.special.gamma(2*self.H+1)
            initial = np.random.normal(0, np.sqrt(variance))

        delta = 1.0 * self.T / n
        self.noise(n)
        f_BM = self.noise

        realisation = [initial]
        t = 0
        
        for k in range(n):
            t += delta 
            initial += ( 
                 -((1/self._scale(t)) * initial) * delta + (self._sigma(t)/(self._scale(t)**self.H))*(f_BM[k+1]- f_BM[k])  )
            realisation.append(initial)
        return realisation
    
    def give_noise(self):
        return self.noise
    
    def interpolation_sample(self, n, initial=0.0):
        """
        Implementation of exact solution for fOU ODE once the fractional Brownian Motion is approximated by a linear interpolation
        """
        if self.H <= 1/4: 
            warnings.warn("Interpolation does not approximate fractional BM for H<=1/4")
                
        delta = 1.0 * self.T / n
        f_BM = self.noise

        realisation = [initial]
        t = 0
        
        for k in range(n):
            t += delta 
            initial = initial*np.exp(-(1/self._scale(t))*delta) + (self._sigma(t)/(self._scale(t)**(self.H-1)))*((f_BM[k+1]- f_BM[k])/delta)*(1-np.exp(-delta/self._scale(t)))
            realisation.append(initial)
        return realisation

    def riemann_sum_sample(self, n, initial=0.0): 
        pass 

