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
            eq_file=None,
            eq_type=None,
            kinetics_file=None,
            kinetics_type=None,
            gk_file=None,
            gk_type=None,
            geometry_type=None,
            linear=True,
            local=True,
            gk_code=None,
            ):

        self._float_format = ''

        self.gk_file = gk_file
        self.gk_type = gk_type
        
        self.geometry_type = geometry_type
        self.linear = linear
        self.local = local
        self.gk_code = gk_code
        self.eq_file = eq_file
        self.eq_type = eq_type
        self.kinetics_file = kinetics_file
        self.kinetics_type = kinetics_type

        # Set code output
        self.set_output_code(gk_code)
        
        # Load equilibrium file if it exists
        if self.eq_file is not None:
            self.load_global_eq()
            
        # Load kinetics file if it exists
        if self.kinetics_file is not None:
           self.load_global_kinetics()
           
        # Read gk_file if it exists (not necessarily load it)
        if self.gk_file is not None:
            self.read_gk_file()

        self.base_directory = Path(__file__).dirname()

    def load_global_eq(self,
                       eq_file=None,
                       eq_type=None,
                       ):
        """
        Loads in global equilibrium parameters

        """
        
        if eq_file is not None:
            self.eq_file = eq_file
            
        if eq_type is not None:
            self.eq_type = eq_type

        if self.eq_type is None or self.eq_file is None:
            raise ValueError('Please specify eq_type and eq_file')
        else:
            self.eq = Equilibrium(self.eq_file, self.eq_type)

    def load_global_kinetics(self,
                             kinetics_file=None,
                             kinetics_type=None
                             ):
        """
        Loads in global kinetic profiles
 
        """
        
        if kinetics_file is not None:
            self.kinetics_file = kinetics_file
            
        if kinetics_type is not None:
            self.kinetics_type = kinetics_type

        if self.kinetics_type is None or self.kinetics_file is None:
            raise ValueError('Please specify kinetics_type and kinetics_file')
        else:
            self.kinetics = Kinetics(self.kinetics_file, self.kinetics_type)

    def read_gk_file(self,
                     gk_file=None,
                     gk_type=None
                     ):
        """ 
        Read GK file

        if self has Equilibrium object then it will
        not load the equilibrium parameters into

        """

        if gk_file is not None:
            self.gk_file = gk_file
            
        if gk_type is not None:
            self.gk_type = gk_type

        if self.gk_type is None or self.gk_file is None:
            raise ValueError('Please specify gk_type and gk_file')
        else:
       
            # If equilibrium already loaded then it won't load the input file
            if hasattr(self, 'eq'):
                template = True
            else:
                template = False

            # Better way to select code?
            if self.gk_type == 'GS2':
                gs2.read(self, self.gk_file, template)
                self.gk_code = 'GS2'
                
            elif self.gk_type == 'CGYRO':
                cgyro.read(self, self.gk_file, template)
                self.gk_code = 'CGYRO'

            else:
                raise NotImplementedError(f'Not implements read for gk_type = {self.gk_type}')

    def write_gk_file(self,
                      file_name,
                      template_file=None,
                      ):
        """ 
        Writes single GK input file to file_name

        
        """

        import os

        self.file_name = file_name

        if self.gk_code == 'GS2':

            # Use template file if given
            if template_file is not None:
                gs2.read(self, data_file=template_file, template=True)

            # If no existing GS2 input file, use template
            if not hasattr(self, "gs2_input"):
                data_file = os.path.join(self.base_directory, 'templates', 'input.gs2')
                gs2.read(self, data_file, template=True)

            gs2.write(self, file_name)

        elif self.gk_code == 'CGYRO':

            # Use template file if given
            if template_file is not None:
                gs2.read(self, data_file=template_file, template=True)

            # If no existing CGYRO input file, use template
            if not hasattr(self, "cgyro_input"):
                data_file = os.path.join(self.base_directory, 'templates', 'input.cgyro')
                cgyro.read(self, data_file, template=True)

            cgyro.write(self, file_name)

        else:
            raise NotImplementedError(f'Writing output for {self.gk_code} not yet available')

    def add_flags(self,
                  flags,
                  ):
        """ 
        Adds flags to GK file

        """

        if self.gk_code == 'GS2':
            gs2.add_flags(self, flags)
        elif self.gk_code == 'CGYRO':
            cgyro.add_flags(self, flags)
        else:
            raise NotImplementedError(f'Adding flags for {self.gk_code} not yet available')

    def load_miller(self,
                    psi_n=None,
                    ):
        """ 
        Loads local Miller geometry parameters

        Adds Miller attribute to Pyro
        """

        self.geometry_type = 'Miller'
        
        if psi_n is None:
            raise ValueError('Need a psi_n to load miller')

        if self.eq is None:
            raise ValueError('Please load equilibrium first')

        self.miller = Miller()
        self.miller.load_from_eq(self.eq, psi_n=psi_n)

    def load_local(self,
                   psi_n=None,
                   geometry_type=None,
                   ):
        """ 
        Loads local geometry and kinetic parameters

        Adds specific geometry and speciesLocal attribute to Pyro
        """

        if psi_n is None:
            raise ValueError('Need a psi_n to load miller')

        if self.eq is None:
            raise ValueError('Please load equilibrium first')

        if geometry_type is None:
            if self.geometry_type is None:
                raise ValueError('Please specify geometry_type')
        else:
            self.geometry_type = geometry_type

        # Load geometry data
        if self.geometry_type == 'Miller':
            self.load_miller(psi_n=psi_n)

        else:
            raise NotImplementedError(f'geometry_type = {self.geometry_type} not yet implemented')

        # Load species data
        self.load_local_species(psi_n=psi_n)

    def load_local_species(self,
                           psi_n=None,
                           ):
        """ 
        Loads local species parameters

        Adds load_local_species attribute to Pyro
        """

        from .local_species import LocalSpecies

        if psi_n is None:
            raise ValueError('Need a psi_n to load')

        local_species = LocalSpecies()

        local_species.from_kinetics(self.kinetics, psi_n=psi_n, lref=self.eq.a_minor)

        self.local_species = local_species

    def set_output_code(self,
                        gk_code
                        ):
        """
        Sets the GK code to be used

        """

        supported_codes = ['GS2', 'CGYRO', None]

        if gk_code in supported_codes:
            self.gk_code = gk_code
        else:
            raise NotImplementedError(f'GK code {gk_code} not yet supported')

    @property
    def float_format(self):
        """ Sets float format for input files

        
        """

        return self._float_format

    @float_format.setter
    def float_format(self,
                     value):
        """ Setter for float_format

        """
        
        self._float_format = value

