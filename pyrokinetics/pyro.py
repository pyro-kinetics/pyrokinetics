import numpy as np
from path import Path
from . import numerics
from . import gs2
from . import cgyro
from .equilibrium import Equilibrium
from .kinetics import Kinetics
from .miller import Miller

class Pyro:
    """
    Basic pyro object able to read, write, run, analyse and plot GK data

    """

    def __init__(
            self,
            eqFile=None,
            eqType=None,
            kinFile=None,
            kinType=None,
            gkFile=None,
            gkType=None,
            geoType=None,
            linear=True,
            local=True,
            gkCode=None,
            ):

        self._floatFormat = '.4g'

        self.gkFile = gkFile
        self.gkType = gkType
        
        self.geoType = geoType
        self.linear = linear
        self.local = local
        self.gkCode = gkCode
        self.eqFile = eqFile
        self.eqType = eqType
        self.kinFile = kinFile
        self.kinType = kinType

        # Set code output
        self.setOutputCode(gkCode)
        
        # Load equilibrium file if it exists
        if self.eqFile is not None:
            self.loadGlobalEquilibrium()
            
        # Load kinetics file if it exists
        if self.kinFile is not None:
           self.loadGlobalKinetics()
           
        # Read gkFile if it exists (not necessarily load it)
        if self.gkFile is not None:
            self.readGKFile()

        self.num = numerics.default()

        self.baseDir = Path(__file__).dirname()


    def loadGlobalEquilibrium(self,
                              eqFile=None,
                              eqType=None,
                              ):
        """
        Loads in global equilibrium parameters

        """
        
        if eqFile is not None:
            self.eqFile = eqFile
            
        if eqType is not None:
            self.eqType = eqType

        if self.eqType is None or self.eqFile is None:
            raise ValueError('Please specify eqType and eqFile')
        else:
            self.eq = Equilibrium(self.eqFile, self.eqType)


    def loadGlobalKinetics(self,
                           kinFile=None,
                           kinType=None
                           ):
        """
        Loads in global kinetic profiles
 
        """
        
        if kinFile is not None:
            self.kinFile = kinFile
            
        if kinType is not None:
            self.kinType = kinType

        if self.kinType is None or self.kinFile is None:
            raise ValueError('Please specify kinType and kinFile')
        else:
            self.kin = Kinetics(self.kinFile, self.kinType)


    def readGKFile(self,
                   gkFile=None,
                   gkType=None
                ):
        """ 
        Read GK file

        if self has Equilibrium object then it will
        not load the equilibrium parameters into

        """

        if gkFile is not None:
            self.gkFile = gkFile
            
        if gkType is not None:
            self.gkType = gkType

        if self.gkType is None or self.gkFile is None:
            raise ValueError('Please specify gkType and gkFile')
        else:
       
            # If equilibrium already loaded then it won't load the input file
            if hasattr(self, 'eq'):
                template = True
            else:
                template = False

            # Better way to select code?
            if self.gkType == 'GS2':
                gs2.read(self, self.gkFile, template)
                self.gkCode = 'GS2'
                
            elif self.gkType == 'CGYRO':
                cgyro.read(self, self.gkFile, template)
                self.gkCode = 'CGYRO'

            else:
                raise NotImplementedError(f'Not implements read for gkType = {self.gkType}')

    
    def writeSingle(self,
                    filename,
                    templateFile=None,
                     ):

        """ 
        Writes single GK input file to filename

        
        """


        self.filename = filename

        if self.gkCode == 'GS2':

            # Use template file if given
            if templateFile is not None:
                gs2.read(self, datafile=templateFile, template=True)

            # If no existing GS2 input file, use template
            if not hasattr(self, "gs2in"):
                datafile = self.baseDir+'/templates/input.gs2'
                gs2.read(self, datafile, template=True)

            gs2.write(self, filename)

        elif self.gkCode == 'CGYRO':

            # Use template file if given
            if templateFile is not None:
                gs2.read(self, datafile=templateFile, template=True)

            # If no existing CGYRO input file, use template
            if not hasattr(self, "cgyroin"):
                datafile = self.baseDir+'/templates/input.cgyro'
                cgyro.read(self, datafile, template=True)

            cgyro.write(self, filename)

        else:
            raise NotImplementedError(f'Writing output for {self.gkCode} not yet available')

    def addFlags(self,
                 flags,
                 ):
        """ 
        Adds flags to GK file

        """


        if self.gkCode == 'GS2':
            gs2.addFlags(self, flags)
        elif self.gkCode == 'CGYRO':
            cgyro.addFlags(self, flags)
        else:
            raise NotImplementedError(f'Adding flags for {self.gkCode} not yet available')

    def loadMiller(self,
                    psiN=None,
                    ):
        """ 
        Loads local Miller geometry parameters

        Adds Miller attribute to Pyro
        """

        self.geoType = 'Miller'
        
        if psiN is None:
            raise ValueError('Need a psiN to load miller')

        if self.eq is None:
            raise ValueError('Please load equilibrium first')

        self.mil = Miller()
        self.mil.fromEq(self.eq, psiN=psiN)

        
    def loadLocal(self,
                   psiN=None,
                   geoType=None,
                   ):
        """ 
        Loads local geometry and kinetic parameters

        Adds specific geometry and speciesLocal attribute to Pyro
        """

        if psiN is None:
            raise ValueError('Need a psiN to load miller')

        if self.eq is None:
            raise ValueError('Please load equilibrium first')

        if geoType is None:
            if self.geoType is None:
                raise ValueError('Please specify geoType')
        else:
            self.geoType = geoType

        # Load geometry data
        if self.geoType == 'Miller':
            self.loadMiller(psiN=psiN)

        else:
            raise NotImplementedError(f'geoType = {self.geoType} not yet implemented')

        # Load species data
        self.loadSpeciesLocal(psiN=psiN)

    def loadSpeciesLocal(self,
                     psiN=None,
                     ):
        """ 
        Loads local species parameters

        Adds loadSpeciesLocal attribute to Pyro
        """

        from .speciesLocal import SpeciesLocal

        if psiN is None:
            raise ValueError('Need a psiN to load')

        spLocal = SpeciesLocal()

        spLocal.fromKinetics(self.kin, psiN=psiN, Lref=self.eq.amin)

        self.spLocal = spLocal


    def setOutputCode(self,
                      gkCode
                      ):
        """
        Sets the GK code to be used

        """

        supportedCodes = ['GS2', 'CGYRO', None]

        if gkCode in supportedCodes:
            self.gkCode = gkCode
        else:
            raise NotImplementedError(f'GK code {gkCode} not yet supported')

    @property
    def floatFormat(self):
        """ Sets float format for input files

        
        """

        return self._floatFormat

    @floatFormat.setter
    def floatFormat(self,
                   value):
        """ Setter for floatFormat

        """
        
        self._floatFormat = value

    
