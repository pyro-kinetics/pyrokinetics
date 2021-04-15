
class Species:
    """
    Contains all species data

    Charge
    Mass
    Density
    Temperature
    Rotation
    

    """

    def __init__(
            self,
            spType=None,
            charge=None,
            mass=None,
            dens=None,
            temp=None,
            rot=None,
            rho=None,
            ang=None
            ):

        self.spType = spType
        self.charge = charge
        self.mass = mass
        self.dens = dens
        self.temp = temp
        self.rot = rot
        self.ang = ang
        self.rho = rho
        self.gradrho = self.rho.derivative()

    def getMass(
            self,
            ):

        return self.mass


    def getCharge(
            self,
            ):

        return self.charge

    def getDens(
            self,
            psiN=None
            ):

        return self.dens(psiN)

    def getLn(self,
                       psiN=None
                       ):
        """
        - 1/n dn/psiN
        """
        
        dens = self.getDens(psiN)
        gradn = self.dens.derivative()(psiN)
        gradrho = self.gradrho(psiN)
    
        Ln = -1/dens *  gradn/gradrho

        return Ln
    
    def getTemp(
            self,
            psiN=None
            ):

        return self.temp(psiN)

    def getLT(self,
             psiN=None
             ):

        temp = self.getTemp(psiN)
        gradT = self.temp.derivative()(psiN)
        gradrho = self.gradrho(psiN)
        
        LT = -1/temp * gradT/gradrho

        return LT

    def getVel(self,
               psiN=None
               ):

        if self.rot is None:
            vel = 0.0
        else:
            vel = self.rot(psiN)
            
        return vel
    
    def getLv(self,
               psiN=None
               ):

        vel = self.getVel(psiN)

        if self.rot is None:
            gradv = 0
        else:
            gradv = self.rot.derivative()(psiN)
            
        gradrho = self.gradrho(psiN)

        if vel != 0.0:
            Lv = -1/vel * gradv / gradrho
        else:
            Lv = 0.0

        return Lv
