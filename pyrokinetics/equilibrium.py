from numpy import sqrt
import numpy as np

class Equilibrium:
    """ 
    Partially adapted from equilibrium option developed by
    B. Dudson in FreeGS

    Equilibrium object containing 

    R 
    Z
    Psi(R,Z)
    q (Psi)
    f (Psi)
    ff'(Psi)
    B
    Bp
    Btor
    """

    def __init__(
            self,
            eqFile=None,
            eqType=None,
            ):

        self.eqFile = eqFile
        self.eqType = eqType

        self.nr = None
        self.nz = None
        self.psi = None
        self.psiRZ = None
        self.pressure = None
        self.pprime = None
        self.fpsi = None
        self.ffprime = None
        self.q = None
        self.Bp = None
        self.Bt = None
        self.amin = None
        self.current = None

        if self.eqFile is not None:
            if self.eqType == 'GEQDSK':
                self.readGeqdsk()

            elif self.eqType == None:
                raise ValueError('Please specify the type of equilibrium')
            else:
                raise NotImplementedError(f"Equilibrium type {self.eqType} not yet implemented")


    def BR(self, R, Z):

        BR = -1/R * self.psiRZ(R, Z, dy=1, grid=False)

        return BR

    def BZ(self, R, Z):

        BZ = 1/R * self.psiRZ(R, Z, dx=1, grid=False)

        return BZ

    def Bpol(self, R, Z):

        BR = self.BR(R, Z)
        BZ = self.BZ(R, Z)

        Bpol = sqrt(BR**2 + BZ**2)

        return Bpol

    def Btor(self, R, Z):

        psi = self.psiRZ(R, Z, grid=False)

        psiN = (psi - self.psi_axis)/(self.psi_bdry - self.psi_axis)
        f = self.fpsi(psiN)

        Btor = f / R

        return Btor

    def readGeqdsk(self):
        """

        Read in GEQDSK file and populates Equilibrium object

        """

        from freegs import _geqdsk
        from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
        from numpy import linspace
        
        f = open(self.eqFile)
        gdata = _geqdsk.read(f)
        f.close()
        
        # Assign gdata to Equilibriun object

        self.nr = gdata['nx']
        self.nz = gdata['ny']

        psiRZ = gdata['psi']
        self.bcentr = gdata['bcentr']

        self.psi_axis = gdata['simagx']
        self.psi_bdry = gdata['sibdry']

        psiN = linspace(0.0, 1.0, self.nr)

        # Set up 1D profiles as interpolated functions
        self.fpsi = InterpolatedUnivariateSpline(
            psiN, gdata['fpol'])
        
        self.ffprime = InterpolatedUnivariateSpline(
            psiN, gdata['ffprime'])
            
        self.q = InterpolatedUnivariateSpline(
            psiN, gdata['qpsi'])
            
        self.pressure = InterpolatedUnivariateSpline(
            psiN, gdata['pres'])
            
        self.pprime = self.pressure.derivative()


        # Set up 2D psiRZ grid
        self.R = linspace(gdata['rleft'],
                          gdata['rleft'] + gdata['rdim'], self.nr)

        self.Z = linspace(gdata['zmid'] - gdata['zdim']/2,
                          gdata['zmid'] + gdata['zdim']/2, self.nz)

        self.psiRZ = RectBivariateSpline(self.R, self.Z, psiRZ)

        rho = psiN * 0.0
        Rmaj = psiN * 0.0
        
        for i, i_psiN in enumerate(psiN[1:]):

            surface_R, surface_Z = self.getFluxSurface(psiN=i_psiN)

            rho[i+1] = (max(surface_R) - min(surface_R)) / 2
            Rmaj[i+1] = (max(surface_R) + min(surface_R)) / 2

        self.LCFS_R = surface_R
        self.LCFS_Z = surface_Z

        self.amin = rho[-1]
        
        rho = rho/rho[-1]

        Rmaj[0] = Rmaj[1] + psiN[1] * (Rmaj[2] - Rmaj[1])/(psiN[2] - psiN[1])

        self.rho = InterpolatedUnivariateSpline(
            psiN, rho)

        self.Rmaj = InterpolatedUnivariateSpline(
            psiN, Rmaj)

            
    def getFluxSurface(self,
                       psiN=None,
                       ):

        from numpy import meshgrid, array
        import matplotlib.pyplot as plt
        
        if psiN is None:
            raise ValueError('getFluxSurface needs a psiN')

        # Generate 2D mesh of normalised psi 
        psi2D = np.transpose(self.psiRZ(self.R, self.Z))

        psiN2D = (psi2D - self.psi_axis) / (self.psi_bdry - self.psi_axis)

        # Returns a list of list of contours for psiN
        con = plt.contour(self.R, self.Z, psiN2D, levels=[0, psiN])
        plt.clf()
        
        paths = con.collections[1].get_paths()
        path = paths[np.argmax(len(paths))]
        
        Rcon, Zcon = path.vertices[:, 0], path.vertices[:, 1]

        #Start from OMP
        Zcon = np.flip(np.roll(Zcon, -np.argmax(Rcon)-1))
        Rcon = np.flip(np.roll(Rcon, -np.argmax(Rcon)-1))

        return Rcon, Zcon

    def generateLocal(self,
                       geoType=None,
                       ):
        
        raise NotImplementedError
