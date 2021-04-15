from scipy.interpolate import InterpolatedUnivariateSpline
from .constants import *
import numpy as np
import netCDF4 as nc
from .species import Species

class Kinetics:
    """
    Contains all the kinetic data
    made up of Species object
    
    Dictionary of Species with key being species name

    kinetics['electron'] = Species(electron)

    Need to do this for all species

    """

    def __init__(
            self,
            kinFile=None,
            kinType=None,
            nspec=None
            ):

        self.kinFile = kinFile
        self.kinType = kinType
        self.nspec = nspec

        if self.kinType == 'SCENE':
            self.readSCENE()
        elif self.kinType == 'JETTO':
            self.readJETTO()
        elif self.kinType == 'TRANSP':
            self.readTRANSP()


    def readSCENE(self):
        """

        Read NetCDF file from SCENE code
        Assumes 3 species: e, D, T

        """

        kdata = nc.Dataset(self.kinFile)

        self.nspec = 3
        
        psi = kdata['Psi'][::-1]
        psiN = psi/psi[-1]
        
        rho = kdata['TGLF_RMIN'][::-1]
        rhofunc = InterpolatedUnivariateSpline(psiN, rho)
        
        spData = {}
        
        Tedata = kdata['Te'][::-1]
        Tefunc = InterpolatedUnivariateSpline(psiN, Tedata)

        Nedata = kdata['Ne'][::-1]
        Nefunc = InterpolatedUnivariateSpline(psiN, Nedata)

        Vedata = Tedata * 0.0
        Vefunc = InterpolatedUnivariateSpline(psiN, Vedata)

        electron = Species(spType='electron',
                           charge=-1,
                           mass=mElectron,
                           dens=Nefunc,
                           temp=Tefunc,
                           rot=Vefunc,
                           rho=rhofunc)

        spData['electron'] = electron
        
        Tifunc = Tefunc
        Vifunc = Vefunc

        Nifunc = InterpolatedUnivariateSpline(psiN, Nedata/2)
        
        deuterium = Species(spType='deuterium',
                            charge=1,
                            mass=mDeuterium,
                            dens=Nifunc,
                            temp=Tifunc,
                            rot=Vifunc,
                            rho=rhofunc)

        spData['deuterium'] = deuterium
        
        tritium = Species(spType='tritium',
                          charge=1,
                          mass=1.5*mDeuterium,
                          dens=Nifunc,
                          temp=Tifunc,
                          rot=Vifunc,
                          rho=rhofunc)

        spData['tritium'] = tritium

        self.spData = spData

        self.spName = [*self.spData.keys()]

        
    def readTRANSP(self):
        """ 
        Reads in TRANSP profiles NetCDF file


        """

        
        kdata = nc.Dataset(self.kinFile)
        
        psi = kdata['PLFLX'][-1, :].data
        psi = psi - psi[0]
        psiN = psi/psi[-1]
        
        rho = kdata['RMNMP'][-1, :].data
        rho = rho/rho[-1]
        
        rhofunc = InterpolatedUnivariateSpline(psiN, rho)
        
        spData = {}

        nspec = 1
        
        Tedata = kdata['TE'][-1, :].data
        Tefunc = InterpolatedUnivariateSpline(psiN, Tedata)

        Nedata = kdata['NE'][-1, :].data  * 1e6
        Nefunc = InterpolatedUnivariateSpline(psiN, Nedata)

        Omegaedata = kdata['OMEG_VTR'][-1, :].data
        Omegaefunc = InterpolatedUnivariateSpline(psiN, Omegaedata)

        electron = Species(spType='electron',
                           charge=-1,
                           mass=mElectron,
                           dens=Nefunc,
                           temp=Tefunc,
                           ang=Omegaefunc,
                           rho=rhofunc)

        spData['electron'] = electron

        # TRANSP only has one ion temp
        Tidata = kdata['TI'][-1, :].data
        Tifunc = InterpolatedUnivariateSpline(psiN, Tidata)


        # Go through each species output in TRANSP

        # Deuterium
        try:
            Nddata = kdata['ND'][-1, :].data * 1e6

            nspec += 1
            Ndfunc = InterpolatedUnivariateSpline(psiN, Nddata)

            Omegaddata = Omegaedata
            Omegadfunc = InterpolatedUnivariateSpline(psiN, Omegaddata)
            
            deuterium = Species(spType='deuterium',
                                charge=1,
                                mass=mDeuterium,
                                dens=Ndfunc,
                                temp=Tifunc,
                                ang=Omegadfunc,
                                rho=rhofunc)
            
            spData['deuterium'] = deuterium

        except IndexError:
            pass

        # Tritium
        try:
            Ntdata = kdata['NT'][-1, :].data * 1e6

            nspec += 1
            Ntfunc = InterpolatedUnivariateSpline(psiN, Ntdata)

            Omegatdata = Omegaedata
            Omegatfunc = InterpolatedUnivariateSpline(psiN, Omegatdata)
            
            tritium = Species(spType='tritium',
                              charge=1,
                              mass=1.5*mDeuterium,
                              dens=Ntfunc,
                              temp=Tifunc,
                              ang=Omegatfunc,
                              rho=rhofunc)

            spData['tritium'] = tritium

        except IndexError:
            pass

        # Helium 4
        try:
            Nhe4data = kdata['NI4'][-1, :].data  * 1e6
        
            nspec += 1
            Nhe4func = InterpolatedUnivariateSpline(psiN, Nhe4data)

            Omegahe4data = Omegaedata
            Omegahe4func = InterpolatedUnivariateSpline(psiN, Omegahe4data)
            
            helium = Species(spType='helium',
                              charge=2,
                              mass=4*mHydrogen,
                              dens=Nhe4func,
                              temp=Tifunc,
                              ang=Omegahe4func,
                              rho=rhofunc)

            spData['helium'] = helium
        except IndexError:
            pass

        # Helium 3
        try:
            Nhe3data = kdata['NI3'][-1, :].data * 1e6
        
            nspec += 1
            Nhe3func = InterpolatedUnivariateSpline(psiN, Nhe3data)

            Omegahe3data = Omegaedata
            Omegahe3func = InterpolatedUnivariateSpline(psiN, Omegahe3data)
            
            helium3 = Species(spType='helium3',
                              charge=2,
                              mass=4*mHydrogen,
                              dens=Nhe3func,
                              temp=Tifunc,
                              ang=Omegahe3func,
                              rho=rhofunc)

            spData['helium3'] = helium3
        except IndexError:
            pass

        try:
            Nimpdata = kdata['NIMP'][-1, :].data * 1e6

            nspec += 1
            Nimpfunc = InterpolatedUnivariateSpline(psiN, Nimpdata)

            Omegaimpdata = Omegaedata
            Omegaimpfunc = InterpolatedUnivariateSpline(psiN, Omegaimpdata)
            
            Z = int(kdata['XZIMP'][-1].data)
            M = int(kdata['AIMP'][-1].data)
            
            impurity = Species(spType='impurity',
                              charge=Z,
                              mass=M*mHydrogen,
                              dens=Nimpfunc,
                              temp=Tifunc,
                              ang=Omegaimpfunc,
                              rho=rhofunc)

            spData['impurity'] = impurity

        except IndexError:
            pass
        
        self.spData = spData

        self.spName = [*self.spData.keys()]
        

    def readJETTO(self):
        """ 
        Reads in JETTO profiles NetCDF file
        Loads Kinetics object

        """

        
        kdata = nc.Dataset(self.kinFile)
        
        psi = kdata['PSI'][-1, :].data
        psi = psi - psi[0]
        psiN = psi/psi[-1]
        
        rho = kdata['RMNMP'][-1, :].data
        rhofunc = InterpolatedUnivariateSpline(psiN, rho)
        
        spData = {}

        nspec = 1

        # Electron data
        Tedata = kdata['TE'][-1, :].data
        Tefunc = InterpolatedUnivariateSpline(psiN, Tedata)

        Nedata = kdata['NE'][-1, :].data
        Nefunc = InterpolatedUnivariateSpline(psiN, Nedata)

        Vedata = kdata['VTOR'][-1, :].data
        Vefunc = InterpolatedUnivariateSpline(psiN, Vedata)

        electron = Species(spType='electron',
                           charge=-1,
                           mass=mElectron,
                           dens=Nefunc,
                           temp=Tefunc,
                           rot=Vefunc,
                           rho=rhofunc)

        spData['electron'] = electron

        # JETTO only has one ion temp
        Tidata = kdata['TI'][-1, :].data
        Tifunc = InterpolatedUnivariateSpline(psiN, Tidata)

        # Go through each species output in JETTO

        # Deuterium data
        Nddata = kdata['NID'][-1, :].data
        
        if any(Nddata):
            nspec += 1
            Ndfunc = InterpolatedUnivariateSpline(psiN, Nddata)

            Vddata = Vedata
            Vdfunc = InterpolatedUnivariateSpline(psiN, Vddata)
            
            deuterium = Species(spType='deuterium',
                                charge=1,
                                mass=mDeuterium,
                                dens=Ndfunc,
                                temp=Tifunc,
                                rot=Vdfunc,
                                rho=rhofunc)
            
            spData['deuterium'] = deuterium

        # Tritium data
        Ntdata = kdata['NIT'][-1, :].data

        if any(Ntdata):
            nspec += 1
            Ntfunc = InterpolatedUnivariateSpline(psiN, Ntdata)

            Vtdata = Vedata
            Vtfunc = InterpolatedUnivariateSpline(psiN, Vtdata)
            
            tritium = Species(spType='tritium',
                              charge=1,
                              mass=3*mHydrogen,
                              dens=Ntfunc,
                              temp=Tifunc,
                              rot=Vtfunc,
                              rho=rhofunc)

            spData['tritium'] = tritium

        # Helium data
        Nalpdata = kdata['NALF'][-1, :].data

        if any(Nalpdata):
            nspec += 1
            Nalpfunc = InterpolatedUnivariateSpline(psiN, Nalpdata)

            Valpdata = Vedata
            Valpfunc = InterpolatedUnivariateSpline(psiN, Valpdata)
            
            helium = Species(spType='helium',
                              charge=2,
                              mass=2*mDeuterium,
                              dens=Nalpfunc,
                              temp=Tifunc,
                              rot=Valpfunc,
                              rho=rhofunc)

            spData['helium'] = helium

        # Impurity data
        Nimpdata = kdata['NIMP'][-1, :].data

        if any(Nimpdata):
            nspec += 1
            Nimpfunc = InterpolatedUnivariateSpline(psiN, Nimpdata)

            Vimpdata = Vedata
            Vimpfunc = InterpolatedUnivariateSpline(psiN, Vimpdata)

            Z = int(kdata['ZIA1'][-1, 0].data)
            M = getImpMass(Z)
            
            impurity = Species(spType='impurity',
                              charge=Z,
                              mass=M*mHydrogen,
                              dens=Nimpfunc,
                              temp=Tifunc,
                              rot=Vimpfunc,
                              rho=rhofunc)

            spData['impurity'] = impurity

        self.spData = spData

        self.spName = [*self.spData.keys()]


def getImpMass(Z=None):
    """ Get impurity mass from charge
        
    """
    
    Zlist = [2, 6, 8, 10, 18, 54, 74]
    Mlist = [4, 12, 16, 20, 40, 132, 184]

    M = Mlist[Zlist.index(Z)]
    
    return M
