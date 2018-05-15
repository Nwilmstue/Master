# -*- coding: utf-8 -*-
"""Example of non-lifting potential flow over a cylinder
Example
-------
This example demonstrates the simulation of a two-dimensional
non-lifting potential flow over a cylinder. A rectangular
ambient domain of size L×L is considered, from which a circle of
radius R is trimmed, resulting in the physical domain Ω. Neumann
conditions are imposed on all boundaries in accordance with the
exact solution. A zero-average potential constraint is added
to yield a well-posed boundary value problem::
  Δφ   = 0           in Ω
  ∇φ⋅n = ∇φexact⋅n   on ∂Ω
  ∫φ   = ∫φexact
"""
#! /usr/bin/env python3
from nutils import mesh, plot, cli, function, element, solver, topology, log, numeric
import unittest, numpy
def main(
    uinf      = 2.0,
    L         = 2.0,
    R         = 0.5,
    nelems    = 7  ,
    degree    = 2  ,
    maxrefine = 3  ,
    withplots = True
    ):
  
  """`main` functions with different parameters.
  
  Parameters
  ----------
  uinf : float 
      free stream velocity
  L : float
      domain size
  R : float
      cylinder radius
  nelems : int
      number of elements
  degree : int
      b-spline degree
  maxrefine : int
      bisectioning steps
  withplots : bool
      create plots
  Returns
  -------
  lhs : float
      solution φ
  err : float
      L2 norm, H1 norm and energy errors
  """
  # construct mesh
  verts = numpy.linspace(-L/2, L/2, nelems+1)
  domain, geom = mesh.rectilinear([verts, verts])
  # trim out a circle
  domain = domain.trim(function.norm2(geom)-R, maxrefine=maxrefine)
  # initialize namespace
  ns = function.Namespace()
  ns.R    = R
  ns.x    = geom
  ns.uinf = uinf
  # construct function space and lagrange multiplier
  ns.phibasis, ns.lbasis = function.chain([domain.basis('spline', degree=degree),[1.]])
  ns.phi = ' phibasis_n ?lhs_n'
  ns.u   = 'sqrt( (phibasis_n,i ?lhs_n) (phibasis_m,i ?lhs_m) )'
  ns.l   = 'lbasis_n ?lhs_n'
  # set the exact solution
  ns.phiexact = 'uinf x_1 ( 1 - R^2 / (x_0^2 + x_1^2) )' # average is zero
  ns.phierror = 'phi - phiexact'
  # construct residual
  res  = domain.integral('-phibasis_n,i phi_,i' @ ns, geometry=ns.x, degree=degree*2)
  res += domain.boundary.integral('phibasis_n phiexact_,i n_i' @ ns, geometry=ns.x, degree=degree*2)
  res += domain.integral('lbasis_n phi + l phibasis_n' @ ns, geometry=ns.x, degree=degree*2)
  # find lhs such that res == 0 and substitute this lhs in the namespace
  lhs = solver.solve_linear('lhs', res)
  ns = ns(lhs=lhs)
  # evaluate error
  err1 = numpy.sqrt(domain.integrate(['phierror phierror' @ns,'phierror phierror + phierror_,i phierror_,i' @ns, '0.5 phi_,i phi_,i' @ns], geometry=ns.x, degree=degree*4))
  err2 = abs(err1[2] - 2.7710377946088443)
  err = numpy.array([err1[0],err1[1],err2])
  log.info('errors: L2={:.2e}, H1={:.2e}, eh={:.2e}'.format(*err))
  if withplots:
    makeplots(domain, ns)
  return lhs, err
def makeplots(domain, ns):
  
  # velocity field evaluation
  points, vals = domain.simplex.elem_eval([ns.x, ns.u], ischeme='bezier5', separate=True)
  # trimming curve evaluation
  tpoints = domain.boundary['trimmed'].elem_eval(ns.x, ischeme='bezier5', separate=True)
  # background domain evaluation
  background_domain = topology.UnstructuredTopology(ndims=2, elements=[element.Element(element.LineReference()**2,elem.transform) for elem in domain])
  bpoints = background_domain.simplex.elem_eval(ns.x, ischeme='bezier2', separate=True)
  with plot.PyPlot('cylinder', ndigits=1) as plt:
    plt.mesh(points , vals, edgecolors='none')
    plt.mesh(bpoints, edgecolors='r', edgewidth=1.5, mergetol=1e-6)
    plt.segments( tpoints , color='g', lw=2 )
    plt.gca().axis('off')
    plt.axis('equal') 
def convergence(
    degree    = 2 ,
    nrefine   = 6,
    maxrefine = 7
    ):
  """Performs convergence study for the example
  Parameters
  ----------
  degree    : int
      b-spline degree
  nrefine   : int
      level of h-refinement
  maxrefine : int
      bisectioning steps
  """
  l2err, h1err, errh = numpy.array([main(nelems=2**(1+irefine), maxrefine = nmaxrefine, degree=degree)[1] for nmaxrefine in range(1,maxrefine+1) for irefine in log.range('refine', nrefine)]).T
  h = .5**numpy.arange(nrefine)
  clrstr = ['k*--','ks--','k^--','ko--','k--','k+--','kv--','k<--','kX--','k>--']
  # ploting errors
  with plot.PyPlot('L2error',ndigits=0) as plt:
    for m in range(0,maxrefine):
      plt.loglog(h, l2err[nrefine*m:nrefine*(m+1)], clrstr[m],label='maxrefine ={}'.format(m))
      plt.slope_triangle(h, l2err[nrefine*m:nrefine*(m+1)])
      plt.legend()
    plt.ylabel('L2 error')
    plt.grid(True)
  with plot.PyPlot('H1error',ndigits=0) as plt:
    for m in range(0,maxrefine):
      plt.loglog(h, h1err[nrefine*m:nrefine*(m+1)], clrstr[m],label='maxrefine ={}'.format(m))
      plt.slope_triangle(h, h1err[nrefine*m:nrefine*(m+1)])
      plt.legend()
    plt.ylabel('H1 error')
    plt.grid(True) 
  
  with plot.PyPlot('Energy_error',ndigits=0) as plt:
    for m in range(0,maxrefine):
      plt.loglog(h, errh[nrefine*m:nrefine*(m+1)], clrstr[m],label='maxrefine ={}'.format(m))
      plt.slope_triangle(h, errh[nrefine*m:nrefine*(m+1)])
      plt.legend()
    plt.ylabel('Energy error')
    plt.grid(True) 
class test(unittest.TestCase):
  def test_p1(self) :
    lhs, err = main( nelems=4, degree=1, withplots=False)
    numeric.assert_allclose64(lhs,'eNqrOH7lBAODjnmHxYLjE08yMOSbJVgsOR52hoFhlXEMklgFXB0DAwA7UhP3')
    numpy.testing.assert_almost_equal(err[1], 0.9546445145978546, decimal=6)
  def test_p2(self):
    lhs, err = main(nelems=4, degree=2, withplots=False)
    numeric.assert_allclose64(lhs,'eNqbcNzixLZTXqYnzAsslhxPOhFyao3pPPMYi6vHT56oM2s6aW6ujcJGVjMBSS8DAwCNgyPv')
    numpy.testing.assert_almost_equal(err[1], 0.42218496186521309, decimal=6)
if __name__ == '__main__':
  cli.choose(main, convergence)