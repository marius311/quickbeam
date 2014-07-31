import numpy as np

import quickbeam as qb

class bmap(object):
    def __init__(self, mapdat, nx, ny, xcentre, ycentre, deltax, deltay):
        self.mapdat  = mapdat
        
        self.nx      = nx
        self.ny      = ny
        self.xcentre = xcentre
        self.ycentre = ycentre
        self.deltax  = deltax
        self.deltay  = deltay

def read_fits_bmap(tfname, mapfield='Beamdata', hdu=1, fortran=False):
    import pyfits

    offset = { True : 1, False : 0 }[fortran]

    hdu  = pyfits.open(tfname)[hdu]

    nx      = hdu.header['Nx']
    ny      = hdu.header['Ny']
    deltax  = hdu.header['Xdelta']
    deltay  = hdu.header['Ydelta']
    xcentre = hdu.header['Xcentre'] - offset
    ycentre = hdu.header['Ycentre'] - offset
    
    mapdat  = hdu.data.field(mapfield).reshape( nx, ny )

    return bmap(mapdat, nx, ny, xcentre, ycentre, deltax, deltay)

def write_fits_bmap(tfname, tbmap, mapfield='Beamdata', fortran=False):
    import pyfits

    offset = { True : 1, False : 0 }[fortran]

    hdu     = pyfits.new_table( [pyfits.Column(name=mapfield,  format=r'1E', unit='[arbitrary]', array=tbmap.mapdat.ravel()) ] )

    hdu.header.update('HIERARCH Nx',      tbmap.nx)
    hdu.header.update('HIERARCH Ny',      tbmap.ny)
    hdu.header.update('HIERARCH Xdelta',  tbmap.deltax)
    hdu.header.update('HIERARCH Ydelta',  tbmap.deltay)
    hdu.header.update('HIERARCH Xcentre', tbmap.xcentre + offset)
    hdu.header.update('HIERARCH Ycentre', tbmap.ycentre + offset)
    hdu.name = 'xtension'

    hdu.writeto(tfname)

def read_dpc_bmap(obj, fortran=False):
    import piolib

    offset = { True : 1, False : 0 }[fortran]

    print 'reading ', obj

    nx      = piolib.ReadKeywordObject("Nx",      "PIOINT",    obj)[0]
    ny      = piolib.ReadKeywordObject("Ny",      "PIOINT",    obj)[0]
    deltax  = piolib.ReadKeywordObject("Xdelta",  "PIODOUBLE", obj)[0]
    deltay  = piolib.ReadKeywordObject("Ydelta",  "PIODOUBLE", obj)[0]
    xcentre = piolib.ReadKeywordObject("Xcentre", "PIOINT",    obj)[0] - offset
    ycentre = piolib.ReadKeywordObject("Ycentre", "PIOINT",    obj)[0] - offset

    mapdat  = piolib.ReadVECTObject(obj, "PIOFLOAT", "begin=0;end=%d" % (nx*ny - 1)).reshape( nx, ny )

    return bmap(mapdat, nx, ny, xcentre, ycentre, deltax, deltay)

def write_dpc_bmap(obj, tbmap, fortran=False):
    import piolib

    offset = { True : 1, False : 0 }[fortran]

    print "  writing " + obj

    assert( tbmap.xcentre == int(np.ceil(float(tbmap.nx) / 2)) )
    assert( tbmap.ycentre == int(np.ceil(float(tbmap.ny) / 2)) )

    if (piolib.CheckObject(obj)):
        piolib.DeleteObject(obj)
        
    piolib.CreateVECTObject(obj, "PIOFLOAT")
    piolib.WriteVECTObject( tbmap.mapdat, obj, "PIOFLOAT", "begin=0;end=%d" % (len(tmap.ravel()) - 1))
    piolib.WriteKeywordObject("square",       "injected by quickbeam/bmap.py",  "GridType", "PIOSTRING", obj)
    piolib.WriteKeywordObject(tbmap.deltax,   "grid X step [radians]",          "Xdelta",   "PIODOUBLE", obj)
    piolib.WriteKeywordObject(tbmap.deltay,   "grid Y step [radians]",          "Ydelta",   "PIODOUBLE", obj)
    piolib.WriteKeywordObject(tbmap.nx,       "grid X size",                    "Nx",       "PIOINT",    obj)
    piolib.WriteKeywordObject(tbmap.ny,       "grid Y size",                    "Ny",       "PIOINT",    obj)
    piolib.WriteKeywordObject(tbmap.xcentre + offset,  "centre location (X index)",      "Xcentre",  "PIOINT",    obj)
    piolib.WriteKeywordObject(tbmap.ycentre + offset,  "centre location (Y index)",      "Ycentre",  "PIOINT",    obj)

def bmap2blm( lmax, mmax, nthts, nphis, bmap, normalize=False, k=1, thtmax=None ):
    # asserts may not be necessary,
    # but code has not been properly
    # checked for the case when they
    # are broken.
    assert( bmap.nx == bmap.ny )
    assert( bmap.xcentre == bmap.ycentre )
    assert( bmap.deltax == bmap.deltay )
    
    import scipy.interpolate
    
    xs = (np.arange(0, bmap.nx) - bmap.xcentre) * bmap.deltax
    ys = (np.arange(0, bmap.ny) - bmap.ycentre) * bmap.deltay

    # note: levels/beam2alm uses thtmax = np.arcsin( 0.5 * bmap.deltax * (bmap.nx-1) )
    if thtmax == None:
        thtmax = np.min( [ abs(xs[0]), abs(ys[0]), abs(xs[-1]), abs(ys[-1]) ] )
    print 'bmap2blm: thtmax = ', thtmax * 180.*60./np.pi, 'arcmin'

    interpolator = scipy.interpolate.RectBivariateSpline( xs, ys, bmap.mapdat.T, kx=k, ky=k )

    dtht_interp = thtmax / (nthts-1.) * np.ones( nthts)
    thts_interp = np.arange(0, nthts) * dtht_interp

    dphi_interp = 2.*np.pi / nphis * np.ones(nphis)
    phis_interp = np.arange(0, nphis) * dphi_interp

    phis_interp_grid, thts_interp_grid = np.meshgrid( phis_interp, thts_interp )
    thts_interp_grid = thts_interp_grid.ravel()
    phis_interp_grid = phis_interp_grid.ravel()

    bmap_interp = interpolator.ev( thts_interp_grid * np.cos(phis_interp_grid), thts_interp_grid * np.sin(phis_interp_grid) )
    
    ret = qb.tpg2blm( lmax, mmax, thts_interp, phis_interp, dtht_interp, dphi_interp, bmap_interp.reshape(nthts, nphis) )

    if normalize == True:
        ret[:] /= ret[0,0]
        
    for m in xrange(1, mmax+1):
        ret[:,m] *= (np.cos(m*np.pi*0.5) - np.sin(m*np.pi*0.5)*1.j)
        # FIXME: sign/rotation to be resolved! current set of parameters
        #        gives agreement w/ FICSBell for test_arrowbeam (see also
        #        ghb.ghb2blm).

    return ret
