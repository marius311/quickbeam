# -*- coding: utf-8 -*-

#####################################################################
#####################################################################
# Some tools for Gauss-Hermite recomposition of beams
#
# original code from Kevin Huffenberger
# pythonized by Brendan Crill
# 15 Feb 2010
# revisited by Eric Hivon July 2010
#  bug correction in shapelet_precompute_normHermite_coef
#  (use double variable 'n' instead of integer)
#
# 25 Oct 2010
#  - 200 time speed-up of map computation
#    by Karim Benabed with loop vectorization
#  - issues warning for non-even grid size
#
# 19 May 2011
# - Adding Ruze functionality as a function that calculates beam solid angles:
# 	* ruzeval(theta,freq,lc,rms) - calculates a Ruze envelope
# 	* ghcnr() - creates a GH+Ruze frankenbeam
# 	* beamsa() - calculates the beam solid angle of an input map 
#               (be careful about using sufficient resolution)
#
# here G_n(x) = H_n(x) / sqrt(n!)
# where H_n is the 'physicist' definition of the Hermite polynomial
# (see Wikipedia)
#

import numpy
import math
speedoflight = 299792458.000  # speed of light in m/s

def memoize(fnc):
  cache = {}
  def nf(*args):
    if args in cache:
      return cache[args]
    else:
      r = fnc(*args)
      cache[args] = r
      return r
  return nf
  
def shapelet_cart_idx(n1,n2):
  return (n1 + n2) * ((n1 + n2) + 1) / 2 + n2


def shapelet_cart_2idx(idx):
  na = int((numpy.sqrt(8 * idx + 1) - 1) / 2)
  nb = idx - shapelet_cart_idx(na, 0)
  return na-nb, nb


def shap_Ncoef(nmax):
  return shapelet_cart_idx(nmax+1, 0)


def shap_weight(fwhm_mean):
  sig = fwhm_mean/numpy.sqrt(8.0 * numpy.log(2.0))
  return 1.0 / sig**2 / numpy.pi


def shapelet_cart(x, y, Gx, Gy, n1, n2):
  nn = numpy.double((n1 + n2) / 2.0)
  return Gx[n1] * Gy[n2] * numpy.exp(-(x**2 + y**2) / 2) / (2**nn)


@memoize
def shapelet_precompute_normHermite_coef(nmax):
  C1 = [0.0, 1.0]
  C2 = [0.0, 0.0]
  for n in numpy.arange(2, nmax+1, dtype='float64'):
    C1.append(2.0 / numpy.sqrt(n))
    C2.append(2.0 * numpy.sqrt((n-1) / n))
  return numpy.array(C1), numpy.array(C2)


def shapelet_normHermite(x, C1, C2, nmax):
  G = numpy.zeros((nmax+1,len(x)), dtype='float64')
  G[0] = numpy.ones(len(x))
  if (nmax > 0):
    G[1] = 2.0 * x
    for n in numpy.arange(2, nmax+1):
      G[n] = (C1[n] * x * G[n-1] - C2[n] * G[n-2])
  return G


@memoize
def precomp_shapelet_cart_2idx(nshap):
  nn1 = numpy.zeros(nshap+1, dtype='int')
  nn2 = numpy.zeros(nshap+1, dtype='int')
  for n in numpy.arange(0, nshap+1):
    n1,n2 = shapelet_cart_2idx(n)
    nn1[n] = n1
    nn2[n] = n2
  return nn1, nn2


def shapelet_cart_nogauss(Gx, Gy, n1, n2):
  nn = ((n1+n2) / 2.0)
  return Gx[n1] * Gy[n2] / (2**nn[:,numpy.newaxis])


def beam_shapelets_set_and_sum(rx, ry, beamx, beamy, angle, fwhm_mean, ellip, shap_coeff, nmax):
  nshap = shap_Ncoef(nmax)
  C1, C2 = shapelet_precompute_normHermite_coef(nmax)
  cosangle = numpy.cos(angle * numpy.pi / 180.0)
  sinangle = numpy.sin(angle * numpy.pi / 180.0)
  fwhm_x = fwhm_mean / numpy.sqrt(ellip)
  sig_x  = fwhm_x    / numpy.sqrt(8.0 * numpy.log(2.0))
  sig_y  = sig_x * ellip

  val = numpy.zeros((numpy.size(rx)), dtype='float64')

  u = rx - beamx
  v = ry - beamy

  x = ( cosangle * u + sinangle * v) / sig_x
  y = (-sinangle * u + cosangle * v) / sig_y

  nn1, nn2 = precomp_shapelet_cart_2idx(nshap)
  xy = numpy.concatenate((x,y))
  Gxy = shapelet_normHermite(xy, C1, C2, nmax)
  Gx = Gxy[:,:len(x)]
  Gy = Gxy[:,len(x):]
  val = numpy.sum(shap_coeff[:,numpy.newaxis] * shapelet_cart_nogauss(Gx, Gy, nn1[:-1], nn2[:-1]),0) * numpy.exp(-(x**2 + y**2)/2.)
  return val


def beam_shapelets_map(xmin,xmax,ymin,ymax,nx,ny,beamx,beamy,angle,fwhm_mean,ellip,shapcoeff,nmax):
  """beam_map = beam_shapelets_map(xmin,xmax,ymin,ymax,nx,ny,beamx,beamy,angle,fwhm_mean,ellip,shapcoeff,nmax)
Converts the EGH coefficients in shapcoeff to a cartesian beam map"""

  if (nx%2 != 0 or ny%2 != 0):
    print 'WARNING: better results are obtained when Nx and Ny are even!'
    print 'current values (Nx, Ny) = (%s, %s)' %(nx,ny)
    
  outmap = numpy.zeros((nx, ny), dtype='float64')
  # Replaced arange with linspace to ensure number of elements, 2011-10-04 -RK
  # added endpoint=False to force conformity with the arange vectors 2011-10-19 -RK
  # Similar change done to ghcnr
  #x_step = (xmax - xmin) / nx
  #y_step = (ymax - ymin) / ny
  #xvals = numpy.arange(xmin, xmax, x_step)
  #yvals = numpy.arange(ymin, ymax, y_step)
  xvals = numpy.linspace(xmin, xmax, nx, endpoint=False)
  yvals = numpy.linspace(ymin, ymax, ny, endpoint=False)

  for ix, x in enumerate(xvals):
    #xx = numpy.ones(nx, dtype='float64') * x # wrong
    xx = numpy.ones(ny, dtype='float64') * x # corrected 2011-06-28
    outmap[ix,:] = beam_shapelets_set_and_sum(xx, yvals, beamx, beamy, angle, fwhm_mean, ellip, shapcoeff, nmax)

  return outmap
  
def beam_shapelets_phithetamap(maxtheta,nphi,ntheta, beamx,beamy,angle,fwhm_mean,ellip,shapcoeff,nmax):
  """beam_map = beam_shapelets_phithetamap(maxtheta,nphi,ntheta, beamx,beamy,angle,fwhm_mean,ellip,shapcoeff,nmax)
Converts the EGH coefficients in shapcoeff to a (phi,theta) beam map"""

  outmap    = numpy.zeros((nphi, ntheta), dtype='float64')
  phivals   = numpy.linspace(0., 2.*numpy.pi, nphi,   endpoint=False) # in radians
  thetavals = numpy.linspace(0., maxtheta,    ntheta, endpoint=True) # in arcmin

  for iphi, phi in enumerate(phivals):
    xx = thetavals * numpy.cos(phi)
    yy = thetavals * numpy.sin(phi)
    outmap[iphi,:] = beam_shapelets_set_and_sum(xx, yy, beamx, beamy, angle, fwhm_mean, ellip, shapcoeff, nmax)

  return outmap

def ghcnr(xmin,xmax,ymin,ymax,nx,ny,beamx,beamy,angle,fwhm_mean,ellip,shapcoeff,nmax,freq,lc,rms):
  # The input correlation length (lc) and rms error should be in units of [mm] and [um] respectively
  lc_meter = lc*1e-3
  rms_meter = rms*1e-6
      
  xvals = numpy.linspace(xmin, xmax, nx, endpoint=False)
  yvals = numpy.linspace(ymin, ymax, ny, endpoint=False)
  
  [X,Y] = numpy.meshgrid(xvals,yvals)

  dx = numpy.median(numpy.diff(xvals))			# The resolution of the map
  dy = numpy.median(numpy.diff(yvals))			# The resolution of the map
  
  if (dx != dy):
    print 'WARNING: the step size should be the same for X and Y axis'    
  
  dA = dx*dy							        # The area of a pixel  
  ghmap = beam_shapelets_map(xmin,xmax,ymin,ymax,nx,ny,beamx,beamy,angle,fwhm_mean,ellip,shapcoeff,nmax)

  # calculating the forward gain of the Gauss-Hermite map    
  fg = 4.0*math.pi/arminsq2srad(beamsa(ghmap,dA))

  lamb = speedoflight/freq
  exprho2 = numpy.prod(numpy.exp(-(4.0*math.pi*rms_meter/lamb)**2))	# the loss in forward gain  
  
  shapcoeff_rs = shapcoeff/exprho2				# rescaling the GHC coefficients
  
  # recalculating the GH map
  ghmap = beam_shapelets_map(xmin,xmax,ymin,ymax,nx,ny,beamx,beamy,angle,fwhm_mean,ellip,shapcoeff_rs,nmax)		
  ghvec = ghmap.flatten()
  maxmap = ghvec.max()							# maximum value in map
  ghmap = ghmap/maxmap							# normalizing map
  theta = numpy.sqrt((X-beamx)**2+(Y-beamy)**2)	# the radial distance from center  
  [ruze, gain] = ruzeval(theta,freq,lc_meter,rms_meter)		# Calculating the Ruze envelope
  map = ghmap*(gain*fg)+ruze 					# combining GH map and Ruze envelope
  ruzevec = ruze.flatten()
  maxruze = ruzevec.max()
  map = map/(gain*fg+maxruze)
  map = map*maxmap								# rescaling the map

  return map
  
def ruzeval(theta,freq,lc,rms):
  #RUZEVAL ruze envelope evaluation
  # This function calculates a ruze envelope map taking as input the theta
  # values, i.e. the longitudinal distance on the surface of a sphere away 
  # from the center of a map. Normally Theta is zero in the center of the
  # map. The output is a 2D map that can be added to a beam map along with
  # the gain loss due to Ruze scattering.
  #
  # ruzeval(Theta,freq,lc,rms) - where freq is the frequency in units of Hz,
  # lc is the correlation length in units of m, and rms is the corresponding
  # surace error, also in units of m.% 

  nmax = 10
  lamb = speedoflight/freq

  gain = 1
  map = numpy.zeros(theta.shape,dtype='float32')  
  
  for i in range(4):
     rho = 4.0*math.pi*rms[i]/lamb      
     rho2 = rho**2
     ep = numpy.exp(-rho2)
     env = (2.0*math.pi*lc[i]/lamb)**2
     env = ep*env
     gain = gain*ep

     for n in range(1,nmax):
        delta = rho2**n/(n*math.factorial(n)) * \
           numpy.exp(-(1.0/n)*(math.pi*lc[i]*numpy.sin(theta/60.0/180*math.pi)/lamb)**2)
        map = map+env*delta      
  
  return map,gain
  
def beamsa(map,dA):
  
  maxMap = map.max()
  bsa = numpy.sum(map)*dA/maxMap

  return bsa  
  
def arminsq2srad(arcminsq):
  conv = (360.*360./math.pi/4.0/math.pi)*3600.0  
  srad = arcminsq/conv
  return srad    
  
#def chafreq(channel)
# 
#  return freq
