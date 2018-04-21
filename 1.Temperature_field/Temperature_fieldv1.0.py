# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:34:50 2018

@author: s120803
"""

from nutils import cli, mesh, function, plot, log, solver
import numpy, unittest

def main(
    L: 'Length of container'        = 4,
    H: 'Height of the container'    = 1,
    D: 'Depth of Container'         = 2,
    E: "young's modulus"            = 1e5,
    nu: "poisson's ratio"           = 0.3,
    T: 'far field traction'         = 10,
    nr: 'number of h-refinements'   = 2,
    figures: 'create figures'       = True,
  ):
# Generieke teringzooi    
    ns = function.Namespace()
    domain0, geom = mesh.rectilinear( [L,H,D] )         #Only insert integers here
    ns.x = geom
    print(domain0)
  
    # create the second-order B-spline basis over the coarsest domain
    ns.bsplinebasis = domain0.basis('spline', degree=3)
    ns.controlweights = numpy.ones(len(ns.bsplinebasis))
    ns.weightfunc = 'bsplinebasis_n controlweights_n'
    ns.nurbsbasis = ns.bsplinebasis * ns.controlweights / ns.weightfunc
    
    # create the isogeometric map
    ns.controlpoints = [0,0,0],[0,]
    ns.x_i = 'nurbsbasis_n controlpoints_ni'
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print('Finished')

main()