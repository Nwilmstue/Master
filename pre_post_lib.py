#! /usr/bin/env python3

## @package testtest
# All the functions decicated to the pre and post processing of the simulation are packedf into this library


from nutils import plot, config, parallel, log  
from prettytable import PrettyTable
import numpy, datetime, os, shutil, matplotlib, parmap, tqdm, sys

class figures():
    ## Plot figures in an efficient fashion
    #   @param         ns = The namespace (Namspace class)
    #   @param         prop = All the properties of the simulation (Object properties)
    #   @param         domain = The domain of the problem (topology class)
    #   @param         variable = All variables that needed to be plotted (list op namespace arrays)
    #   @param         name = names of plots (lists of strings)    
    #   @param         clim = colorbar limits (list of lists)  
    #   @param         dpi = Dots per inch in figure (integer)
    #   @param         ii = Layer that is evaluated (integer)   
    def __init__(self,ns,prop,domain,variable,name,clim,dpi, ii):
        assert len(variable) == len(name), 'The length of the variables and name do not match'
        assert len(variable) == len(clim), 'The length of the variables and clim do not match'
        
        self.variable = variable
        self.name = name
        self.ii = ii
        self.dpi = dpi
        self.prop = prop
        self.clim = clim
        self.lenkoriginal = len(variable)
        self.leni = len(variable[0])

        Dinput = [ns.x]              
        if prop.outputfile == 'vtk':
            self.lenk = len(variable) - 1 
            
        for k in range(0,self.lenk):
            for i in range(0,self.leni):
                Dinput.append(variable[k][i][0])
                
        if prop.outputfile == 'vtk':
            self.Total_eval = domain.elem_eval(Dinput, ischeme='vtk', separate=True)
        else:
            self.Total_eval = domain.elem_eval(Dinput, ischeme='vertex1', separate=False)

        if prop.outputfile == 'vtk':
            self.Total_eval = self.Total_eval +tuple(variable[1])


    def printfig(self,name):
        assert name == 'vtk' or name == 'png', 'Not a correct type file, choose vtk or png '
        with config(verbose = 2, nprocs = 8, richoutput = True):
            for i in  parallel.pariter(range(0,self.leni),8):
                for k in range(0,self.lenkoriginal):
                    if name == 'vtk' :
                        self.plotvtk(k,i)
                    else:
                        self.plotpng(k,i)

    def plotpng(self,k,i):
        with plot.PyPlot(self.name[k] + '{}'.format("%04d", self.ii) + '{}'.format("%04d", i)) as plt:
            plt.title(self.name[k] + ' at t={:5.1f}'.format(i*self.prop.timestep))
            plt.mesh(self.Total_eval[0], self.Total_eval[i+1+k*self.leni])
            plt.colorbar(orientation = 'horizontal')
            plt.clim(self.clim[k][0],self.clim[k][1])
            plt.ylabel('Height in [m]')
            plt.xlabel('Width in [m]')
            plt.savefig(self.prop.dir + '2. Figures/' + self.name[k] + str(self.ii) + ' {}'.format(i),bbox_inches='tight', dpi = self.prop.dpi)
    
    def plotvtk(self,k,i):
       with plot.VTKFile(self.name[k] + '{:04d}'.format(self.ii) + '{:04d}'.format(i) ) as vtk:
            vtk.unstructuredgrid(self.Total_eval[0])
            vtk.pointdataarray(self.name[k],self.Total_eval[i+1+k*self.leni])
            
        
## Properties that are used for the simulation
class properties():

    def __init__(self,printprop = True,extended = True, EXIT = False):
        self.TOTAL = []
        #Choices
        self.TOTAL.append(['figures'                    ,True       ,'Create figures'                           , 'Simulation options'])
        self.TOTAL.append(['dpi'                        ,150        ,'Resolution of the figures'                , 'Simulation options'])
        self.TOTAL.append(['adiabatic'                  ,True       ,'Apply adiabatic boundary conditions'      , 'Simulation options'])
        self.TOTAL.append(['Utest'                      ,False      ,'Run Unittest for lhs and constrains'      , 'Simulation options'])
        self.TOTAL.append(['breakvalue'                 ,1000       ,'Until which layer is simulated'           , 'Simulation options'])        
        self.TOTAL.append(['Afig'                       ,5          ,'Amount of figures in a layer  '           , 'Simulation options'])        
        self.TOTAL.append(['outputfile'                ,'vtk'        ,'VTK plots or png'                         , 'Simulation options'])            #werkt nog niet

        #FEM options
        self.TOTAL.append(['ischeme'                    ,'gauss4'   ,'The scheme used for numerical integration'      , 'FEM options'])
        self.TOTAL.append(['basistype'                  ,'spline'   ,'The basisfunctions used over the domain. '      , 'FEM options'])
        self.TOTAL.append(['degree'                     ,3          ,'The polynomial degree of the basisfunctions'    , 'FEM options'])

        # Time solver options
        self.TOTAL.append(['timestep'                   ,1/500        ,'Timestel in [s]'    , 'Time options'])
        self.TOTAL.append(['maxiter'                    ,30           ,'amount of steps of the laser per layer '    , 'Time options'])


        # Material properties
        self.TOTAL.append(['c'                          ,520        ,'Specific heat capacity '                           , 'Material'])
        self.TOTAL.append(['rho'                        ,4510       ,'density '                                          , 'Material'])
        self.TOTAL.append(['k'                          ,16         ,'Heat conductivity '                                , 'Material'])
        self.TOTAL.append(['L'                          ,0          ,'Latent heat production '                           , 'Material'])
        self.TOTAL.append(['Tl'                         ,2000       ,'Temperature where mateirial is liquid '            , 'Material'])
        self.TOTAL.append(['Ts'                         ,1500       ,'Temperature where material is solid'               , 'Material'])
        self.TOTAL.append(['S'                          ,0.2          ,'The strength of TANH curve for phase transistion'  , 'Material'])
        self.TOTAL.append(['h'                          ,200        ,'Convective heat loss'                              , 'Material'])
        
        
        #Heat source (LASER) properties
        self.TOTAL.append(['power'                      ,1.8e8        ,'Laser power'                         , 'Laser properties'])
        
        #Boundary conditions
        self.TOTAL.append(['Toutside'                   ,0          ,'Temperature at infinity'                      , 'Boundary conditions'])
        self.TOTAL.append(['qadiabatic'                 ,0          ,'heat flux when adiabatic'                     , 'Boundary conditions'])

        #initial conditions
        self.TOTAL.append(['EQinitialize'               ,0          ,'Added heat from the start'                    , 'Initial conditions'])
        self.TOTAL.append(['Ti'                         ,200        ,'Intitial Temperature '                        , 'Initial conditions'])
        self.TOTAL.append(['itime'                      ,0          ,'Intitial timeset     '                        , 'Initial conditions'])

        #Geometry and topology variables
        self.TOTAL.append(['LayerResolution'            ,2          ,'Amount of elements per layer'         , 'Geometry and topology'])
        self.TOTAL.append(['n'                          ,10         ,'Amount of layers'                     , 'Geometry and topology'])
        self.TOTAL.append(['ex1'                        ,4          ,'Elements in x1 direction'             , 'Geometry and topology'])
        self.TOTAL.append(['dx1'                        ,0.02       ,'Distance in x1 direction in [m]'      , 'Geometry and topology'])
        self.TOTAL.append(['dx2'                        ,0.005      ,'Distance in x2 direction in [m]'      , 'Geometry and topology'])

        # Create the variables
        for item in range(0,len(self.TOTAL)):
            sol = self.TOTAL[item]
            setattr(self, sol[0], sol[1])

        if printprop:
            self.printprop(extended = True, EXIT = EXIT)
        
        self.dir = ' ' 

    def printprop(self,extended = True, EXIT = False):
        t = PrettyTable(['Variable','Value','Explanation', 'Variable type'])
        sol = self.TOTAL[0]
        typecheck = sol[3]
        for item in range(0,len(self.TOTAL)):
            sol = self.TOTAL[item]
            if not typecheck == sol[3]:
                t.add_row(['      ','      ','      ','      '])
            t.add_row([sol[0], sol[1] ,sol[2], sol[3]])
            typecheck = sol[3]
        log.user(t)

#        log.Log.write(3 , t )

        if extended:
            t = PrettyTable(['What','Value','Explanation'])
            t.add_row(['Layerthickness', self.dx2/self.n ,'The thickness of the layer in [m]'])
            log.user(t)

        if EXIT:
            log.info('Exitted the script in printproperties' )
            sys.exit()
            
    ## Create backup and new directories for figures
    def backup(self):     # Create logfiles and backups
        today = datetime.date.today()
        self.dir = '/home/niki/Documents/1_Master/0_Master_thesis/3_code/Auto/'+str(today) + '/'
        srcfile1 = '/home/niki/Documents/1_Master/0_Master_thesis/3_code/'+str(os.path.basename(__file__))        
        srcfile2 = '/home/niki/Documents/1_Master/0_Master_thesis/3_code/pre_post_lib.py' 
        srcfile3 = '/home/niki/Documents/1_Master/0_Master_thesis/3_code/IGA_AM_lib.py'      
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            os.makedirs(self.dir + '2. Figures/')
        shutil.copy(srcfile1, self.dir)
        shutil.copy(srcfile2, self.dir)
        shutil.copy(srcfile3, self.dir)
        log.info('Backup made')
