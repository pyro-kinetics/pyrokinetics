from .equilibrium import Equilibrium
from collections import OrderedDict
import numpy as np
from scipy.optimize import least_squares
from .constants import *

class Miller(OrderedDict):
    """
    Miller Object representing local Miller fit parameters

    Data stored in a ordereded dictionary

    """
    def __init__(self,
                 *args, **kwargs):

         
        s_args = list(args)
        
        if (args and not isinstance(args[0], OrderedDict)
            and isinstance(args[0], dict)):
            s_args[0] = sorted(args[0].items())
                    
        super(Miller, self).__init__(*s_args, **kwargs)
            


    def fromEq(self,
               eq,
               psiN=None
               ):
        """"
        Loads Miller object from an Equilibrium Object

        """
        
        R, Z = eq.getFluxSurface(psiN=psiN)

        Bpol = eq.Bpol(R, Z)

        Rmaj = eq.Rmaj(psiN)

        Rgeo = Rmaj
        
        rho = eq.rho(psiN)

        rmin = rho * eq.amin
        
        kappa = (max(Z)- min(Z)) / (2*rmin)

        Z0 = (max(Z) + min(Z)) / 2
        
        Zind = np.argmax(abs(Z))
        
        Rupper = R[Zind]
        
        Bgeo = eq.fpsi(psiN)/ Rmaj

        delta = (Rmaj - Rupper)/rmin

        drhodpsi = eq.rho.derivative()(psiN)
        shift = eq.Rmaj.derivative()(psiN)/drhodpsi

        p = eq.pressure(psiN)
        q = eq.q(psiN)

        dqdpsi = eq.q.derivative()(psiN)

        shat = rho/q * dqdpsi/drhodpsi

        dpdrho = eq.pprime(psiN)/drhodpsi

        beta_prime = 8 * pi * 1e-7* dpdrho /Bgeo**2
        
        theta = np.arcsin((Z-Z0)/(kappa*rmin))

        for i in range(len(theta)):
            if R[i] < Rupper:
                if Z[i] >= 0:
                    theta[i] = np.pi - theta[i]
                elif Z[i] < 0:
                    theta[i] = -np.pi - theta[i]
                    
        Rmil, Zmil = millerRZ(theta, kappa, delta, Rmaj, rmin)

        s_kappa_fit = 0.0
        s_delta_fit = 0.0
        shift_fit = shift
        dpsidr_fit = 1.0

        params = [s_kappa_fit, s_delta_fit, shift_fit,
                  dpsidr_fit]

        self['psiN'] = psiN
        self['rho'] = float(rho)
        self['rmin'] = float(rmin)
        self['rmaj'] = float(Rmaj/eq.amin)
        self['rgeo'] = float(Rgeo/eq.amin)
        self['amin'] = float(eq.amin)
        self['Bgeo'] = float(Bgeo)

        self['kappa'] = kappa
        self['delta'] = delta
        self['R'] = R
        self['Z'] = Z
        self['theta'] = theta
        self['Bpol'] = Bpol

        fits = least_squares(self.minBp, params)
        
        self['s_kappa'] = fits.x[0]
        self['s_delta'] = fits.x[1]
        self['shift'] = fits.x[2]
        self['dpsidr'] = fits.x[3]

        self['q'] = float(q)
        self['shat'] = shat
        self['beta_prime'] = beta_prime
        self['pressure'] = p
        self['dpdrho'] = dpdrho

        self['kappri'] = self['s_kappa'] * self['kappa'] / self['rho']
        self['tri'] = np.arcsin(self['delta'])

    
    def minBp(self, params):
        """
        Function for least squares minimisation

        """
        Bp = self.millerBpol(params)

        return self['Bpol'] - Bp

    
    def millerBpol(self, params):
        """
        Returns Miller prediction for Bpol given flux surface parameters

        """

        kappa = self['kappa']
        x = np.arcsin(self['delta'])
        theta = self['theta']
        R = self['R']
        
        s_kappa = params[0]
        s_delta = params[1]
        shift = params[2]
        dpsidr = params[3]
        
        term1 = dpsidr / (kappa * R)
            
        term2 = np.sqrt(np.sin(theta + x * np.sin(theta))**2 *
                        (1 + x * np.cos(theta))**2 + (kappa * np.cos(theta))**2)
        
        term3 = np.cos(x * np.sin(theta)) + shift * np.cos(theta)
            
        term4 = (s_kappa - s_delta*np.cos(theta) + (1 + s_kappa) * x *
                 np.cos(theta)) * np.sin(theta) * np.sin(theta + x * np.sin(theta))
        Bp = term1 * term2 / (term3 + term4)
        
        return Bp


def millerRZ(theta, kappa, delta, rcen, rmin):
    """
    Flux surface given Miller fits
    
    Returns R, Z of flux surface
    """
    R = rcen + rmin * np.cos(theta + np.arcsin(delta)*np.sin(theta))
    Z = kappa * rmin * np.sin(theta)

    return R, Z

def default():
    """
    Default parameters for geometry
    Same as GA-STD case
    """
    
    mil = Miller()

    mil['rho'] = 0.9
    mil['rmin'] = 0.5
    mil['rmaj'] = 3.0
    
    mil['kappa '] = 1.0
    mil['s_kappa'] = 0.0
    mil['kappri'] = 0.0
    
    mil['delta'] = 0.0
    mil['s_delta'] = 0.0
    
    mil['tri'] = 0.0
    mil['tripri'] = 0.0
    
    mil['zeta'] = 0.0
    mil['s_zeta'] = 0.0
    
    mil['q'] = 2.0
    mil['shat'] = 1.0
    
    mil['shift'] = 0.0
    
    mil['btccw'] = -1
    mil['ipccw'] = -1
    
    mil['bprime'] = 0.0
    
    return mil
