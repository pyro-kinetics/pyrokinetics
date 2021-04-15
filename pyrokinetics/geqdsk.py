


class Geqdsk():
    """
    Read in GEQDSK and load into object

    """

    def __init__(self,
                 eqFile=None,
                 ):

        self.eqFile = eqFile
        self.gdata = {}
        
        if self.eqFile is not None:
            self.read()

    def read(self):

        f = open(self.eqFile)

        line = f.readline()

        values = line.split()

        self.gdata['nr'] = int(values[-2])
        self.gdata['nz'] = int(values[-1])

        keys = ['rdim', 'zdim', 'rcentr', 'rleft',
                'zmid', 'rmaxis', 'zmaxis', 'simag',
                'sibry', 'bcentr', 'current', 'simag',
                'xdum', 'rmaxis', 'zdum', 'zmaxis',
                'xdum', 'sibry', 'xdum', 'xdum']

        
        for key in keys():
            self.gdata[key] = float(
    
