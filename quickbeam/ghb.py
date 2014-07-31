# ghb.py
# 
# code for working with Gauss-Hermite beams.

import _ghb
import gauss_hermite

import numpy as np

def ghb2blm(ghb, lmax):
    x0  = ghb.cross_scan
    y0  = ghb.co_scan
    psi = ghb.angle/180.*np.pi
      
    sigma=ghb.fwhm /np.sqrt(8.*np.log(2.))
    sigmax=sigma/np.sqrt(ghb.ellipticity)
    sigmay=sigma*np.sqrt(ghb.ellipticity)

    nspline = (10*2*ghb.nmax+4)
    nphi    = 5*nspline

    blm = _ghb.ghb2balm(lmax, ghb.nmax, ghb.ncoeffs, nspline, nphi, ghb.coeffs, x0, y0, psi, sigmax, sigmay)
    for l in xrange(0,lmax+1):
        blm[l,:] *= np.sqrt(4.*np.pi/(2.*l+1.))

    for m in xrange(1, ghb.nmax+1):
        blm[:,m] *= (np.cos(m*np.pi*0.5) - np.sin(m*np.pi*0.5)*1.j)
    # FIXME: sign/rotation to be resolved! current set of parameters
    #        rotates "y axis" above along the scan direction, though
    #        could be off by 180 deg.

    return blm

## DUMP OF THE RELEVANT PARTS OF "HL2_GaussHermiteBeams/ghb.py", with the intent of using directly in the future.

#####################################################################

import numpy

class ghb:
  #==================================================================
  def __init__(self, nmax=0):
    self.boloid  = ""
    self.nmax    = nmax
    self.ncoeffs = _ghb.cart2indx(self.nmax+1, 0)
    self.coeffs  = numpy.zeros(self.ncoeffs, dtype='float32')
    self.errors  = numpy.zeros(self.ncoeffs, dtype='float32')
    self.covmat  = numpy.zeros(self.ncoeffs * self.ncoeffs, dtype='float32')
    self.cross_scan  = 0.0
    self.co_scan     = 0.0
    self.angle       = 0.0
    self.fwhm        = 0.0
    self.ellipticity = 0.0
    # TODO: implement parameters errors
    # map grid
    self.map    = None	
    self.minx   = 0.0
    self.maxx   = 0.0
    self.miny   = 0.0
    self.maxy   = 0.0
    self.nx     = 0
    self.ny     = 0
    self.mapwr  = None
    self.lc     = numpy.zeros(4, dtype='float32')
    self.rms    = numpy.zeros(4, dtype='float32')
    self.freq   = 0.0
    self.bsa    = 0.0
    self.referential = "unknown"

  #==================================================================
  def compute_map(self, minx, maxx, miny, maxy, nx, ny):
    """ compute_map(minx, maxx, miny, maxy, nx, ny)
     computes Elliptical Gauss Hermite map on cartesian grid
    """
    self.minx = minx
    self.maxx = maxx
    self.miny = miny
    self.maxy = maxy
    self.nx   = nx
    self.ny   = ny
    self.map  = gauss_hermite.beam_shapelets_map(minx, maxx, miny, maxy, nx, ny,
                                                 self.cross_scan,
                                                 self.co_scan,
                                                 self.angle,
                                                 self.fwhm,
                                                 self.ellipticity,
                                                 self.coeffs,
                                                 self.nmax)
												 
  #==================================================================
  def compute_phithetamap(self, maxtheta, nphi, ntheta):
    """ compute_phithetamap(maxtheta, nphi, ntheta)
     computes Elliptical Gauss Hermite map on (phi, theta) grid
    """
    self.maxtheta = maxtheta
    self.nphi     = nphi
    self.ntheta   = ntheta
    self.phithetamap  = gauss_hermite.beam_shapelets_phithetamap(maxtheta, nphi, ntheta,
                                                 self.cross_scan,
                                                 self.co_scan,
                                                 self.angle,
                                                 self.fwhm,
                                                 self.ellipticity,
                                                 self.coeffs,
                                                 self.nmax)

  #==================================================================
  def plot_map(self, minx=None, maxx=None, miny=None, maxy=None, nx=None, ny=None, pngfile=None):
    """plot_map(minx=None, maxx=None, miny=None, maxy=None, nx=None, ny=None, pngfile=None)
    plots the EGH beam map
    """
    import pylab
    
    # update beam map if asked
    if (minx!=None or maxx!=None or miny!=None or maxy!=None or nx!=None or ny!=None):
      if (minx==None or maxx==None or miny==None or maxy==None or nx==None or ny==None):
        raise ValueError("ghb.plot_map(minx, maxx, miny, maxy, nx, ny): you must give all parameters to re-compute the map or none to skip computation")
      self.compute_map(minx, maxx, miny, maxy, nx, ny)

    if (self.map == None) and (self.mapwr == None):
      raise ValueError("ghb.plot_map(): you must call ghb.compute() first, or specify all the parameters of ghb.plot_map()")

    pylab.rc('text', usetex="True")
    ax = pylab.subplot(111) # one row of plots, one column of plots, first plot
    pylab.axes(ax)
    junk = (self.map).transpose() # necessary for contourf to have X along horizontal axis
    lev = (numpy.arange(-1,16,dtype='double'))/16. * junk.max()
    c = ax.contourf(junk, extent=(self.minx, self.maxx, self.miny, self.maxy), levels=lev)
    #     c = pylab.pcolor(self.map)
    #     #c = NonUniformImage(ax,interpolation='nearest', extent=(self.minx, self.maxx, self.miny, self.maxy))
    # 	#c.set_data()
    #     #ax.images.append(c)
    ax.set_xlim((self.minx, self.maxx))
    ax.set_ylim((self.miny, self.maxy))
    if (self.referential == 'Dxx'):
      pylab.xlabel(r'$X_{\rm Dxx}$ (cross-scan) [arcmin]')
      pylab.ylabel(r'$Y_{\rm Dxx}$ (co-scan) [arcmin]')
    if (self.referential == 'Pxx'):
      pylab.xlabel(r'$X_{\rm Pxx}$ (co-polar) [arcmin]')
      pylab.ylabel(r'$Y_{\rm Pxx}$ (cross-polar) [arcmin]')
    pylab.title(self.boloid.replace("_","\_") + r', $n_1+n_2  \le ' + str(self.nmax) + '$')
    pylab.grid()
    pylab.colorbar(c)
    if (pngfile != None):
      pylab.savefig(pngfile)
    else:
      pass
      #pylab.show()
      #pylab.ioff()
    #pylab.close()

  #==================================================================
  def read_from_textfile(self, coeffs_filename, params_filename, boloid, nmax):

    print "Reading ", coeffs_filename
    self.__read_coefficients_file(coeffs_filename, nmax)
    self.__read_parameters_file(params_filename)
    self.boloid = boloid

  ###################################################################
  # private functions
  
  #==================================================================
  def __read_coefficients_file(self, filename, nmax):
    """__read_coefficients_file(filename, nmax)"""
    self.__init__(nmax)
    coeffcount = 0
    matoffset  = 0
    f = open(filename)
    for line in f.readlines():
      line = line.strip(' \n')
      if (line[0] == '#' or len(line) == 0): continue
      cols = line.split()
      if (len(cols) == 2): # value + error
        self.coeffs[coeffcount] = float(cols[0])
        self.errors[coeffcount] = float(cols[1])
        coeffcount += 1
      elif (len(cols) == self.ncoeffs): # covariance matrix
        for i in range(self.ncoeffs):
          self.covmat[matoffset] = float(cols[i])
          matoffset += 1
    file.close(f)
    
    if (len(self.coeffs) != self.ncoeffs): # are we missing some coeffs?
      raise ValueError("ghb.__read_coefficients_file(): bad number of EGH coefficients")
    if (len(self.covmat)  != self.ncoeffs * self.ncoeffs): # are we missing some covariant matrix elements?
      raise ValueError("ghb.__read_coefficients_file(): bad number of EGH covariance matrix elements")


  #==================================================================
  def __read_parameters_file(self, filename):
    """__read_parameters_file(filename)"""
    cols = []
    f = open(filename)
    found_params = False
    for line in f.readlines():
      line = line.strip(' \n')
      if (line[0] == '#' or len(line) == 0): continue
      cols = line.split()
      if (len(cols) == 5):
        self.cross_scan  = float(cols[0])
        self.co_scan     = float(cols[1])
        self.angle       = float(cols[2])
        self.fwhm        = float(cols[3])
        self.ellipticity = float(cols[4])
        found_params = True
    file.close(f)
    if (found_params == False):
      raise ValueError("ghb.__read_parameters_file(): parameters values not found")
