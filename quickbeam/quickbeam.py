import types

import numpy  as np
import healpy as hp

import maths
import healpix

# coefficients for scanning blm -> effective Bl
mwt_hfi_DX7_approx = [ 1.000, 0.133, 0.892, 0.260, 0.757 ] # http://wiki.planck.fr/index.php/Proc/FlatBeams
mwt_hfi_DX8_approx = [ 1.000, 0.074, 0.883, 0.209, 0.737 ]

# --

def bpcl_term(lmax, s1, s2, w12, blm1, blm2, cltt):
    # "beam pseudo-cl term": implementation of Eq.~B3 from
    # arxiv:1003.0198, for fixed s1, s2

    blm1_lmax = blm1.shape[0]-1
    blm2_lmax = blm1.shape[0]-1
    cltt_lmax = len(cltt) - 1
    w12_lmax  = len(w12) - 1

    lmin   = np.max( [abs(s1), abs(s2)] )
    l1max  = w12_lmax
    l2max  = np.min([blm1_lmax, blm2_lmax, cltt_lmax])

    # get gauss legendre points and weights.
    ngl = (l1max + l2max + lmax)/2 + 1
    glq = maths.gauss_legendre_quadrature( int(ngl) )

    # prepare spectra
    twolpo = 2.*np.arange(0, max(l1max, l2max)+1) + 1.

    def blm_pm(blm, l, m):
        if m < 0:
            return (-1)**m * np.conj(blm[l,-m])
        else:
            return blm[l,m]

    blv = np.zeros(l2max+1, dtype=np.complex)
    blv[lmin:(l2max+1)] = [ blm_pm(blm1, l, -s1) * np.conj(blm_pm(blm2, l, -s2)) * cltt[l] for l in xrange(lmin,l2max+1)]

    if s1 == s2 == 0:
        # treat this term (which is generally quite large compared to other terms)
        # as a special case, to avoid numerical issues.
        w12_l0 = w12[0]; w12 = np.copy(w12[:]); w12[0] = 0.0

    # do l sums.
    cf1 = glq.cf_from_cl(  s1,  s2, w12[:]*twolpo[0:(l1max+1)] )
    cf2 = glq.cf_from_cl( -s1, -s2, blv[:]*twolpo[0:(l2max+1)] )

    # do z integral.
    ret  = glq.cl_from_cf(lmax, 0, 0, cf1*cf2)
    ret *= 1./(8.*np.pi)

    if s1 == s2 == 0:
        ret += np.array([blm1[l,0] * blm2[l,0] * cltt[l] * w12_l0 / (4.*np.pi) for l in xrange(0, lmax+1)], dtype=np.complex)
    
    return ret

def bpcl_term_kernel(lmax, s1, s2, w12, blm1, blm2):
    # like bpcl_term, but instead of returning the
    # beam-convolved pseudo-Cl returns the matrix
    # ret[l_out, l_in] such that
    #
    # bpcl_term[l] = \sum_{l=0}^{lmax} ret[l_out, l_in] * cltt[l_in]
    #

    assert( s1 == 0 ); assert( s2 == 0 ) #FIXME: implemented but not yet tested.

    blm1_lmax = blm1.shape[0]-1
    blm2_lmax = blm1.shape[0]-1
    w12_lmax  = len(w12) - 1

    lmin   = np.max( [abs(s1), abs(s2)] )
    l1max  = w12_lmax
    l2max  = np.min([blm1_lmax, blm2_lmax, lmax])

    # prepare spectra
    twolpo = 2.*np.arange(0, max(l1max, l2max)+1) + 1.

    def blm_pm(blm, l, m):
        if m < 0:
            return (-1)**m * np.conj(blm[l,-m])
        else:
            return blm[l,m]

    blv = np.zeros(l2max+1, dtype=np.complex)
    blv[lmin:(l2max+1)] = [ blm_pm(blm1, l, -s1) * np.conj(blm_pm(blm2, l, -s2)) for l in xrange(lmin,l2max+1)]

    if s1 == s2 == 0:
        # treat this term (which is generally quite large compared to other terms)
        # as a special case, to avoid numerical issues.
        w12_l0 = w12[0]; w12 = np.copy(w12[:]); w12[0] = 0.0

    ngl = int( (l1max + l2max + lmax)/2 + 1 )
    zvec, wvec = maths._mathsc.init_gauss_legendre_quadrature(ngl)

    # do l sums.
    cf1 = maths._mathsc.wignerd_cf_from_cl( 0, 0,       1, ngl, l1max, zvec, w12[:].real*twolpo[0:(l1max+1)] )
    cf2 = maths._mathsc.wignerd_cf_from_cl( 0, 0, l2max+1, ngl, l2max, zvec, np.diag( blv.real[:]*twolpo[0:(l2max+1)] ).flatten() )

    if (s1 != 0) or (s2 != 0):
        cf1 = cf1 + maths._mathsc.wignerd_cf_from_cl( 0, 0,       1, ngl, l1max, zvec, w12[:].imag*twolpo[0:(l1max+1)] )*1.j
        cf2 = cf2 + maths._mathsc.wignerd_cf_from_cl( 0, 0, l2max+1, ngl, l2max, zvec, np.diag( blv.real[:]*twolpo[0:(l2max+1)] ).flatten() )*1.j

    # do z integral.
    for i in xrange(0, l2max+1):
        cf2[i*ngl:(i+1)*ngl] *= cf1

    ret = maths._mathsc.wignerd_cl_from_cf( 0, 0, l2max+1, ngl,  lmax, zvec, wvec, cf2.real.flatten() ).reshape( (lmax+1, lmax+1) )

    if (s1 != 0) or (s2 != 0):
        ret = ret + maths._mathsc.wignerd_cl_from_cf( 0, 0, l2max+1, ngl,  lmax, zvec, wvec, cf2.imag.flatten() ).reshape( (lmax+1, lmax+1) )*1.j

    ret *= 1./(8.*np.pi)

    if s1 == s2 == 0:
        for l in xrange(0, lmax+1):
            ret[l,l] += blm1[l,0].real * blm2[l,0].real * w12_l0 / (4.*np.pi)

    ret = ret.T

    return ret

def convolve_s(s, wsmap, blm, alm):
    """ calculates T(p) = sum_{lm (S \in {+s, -s})} w(p, -S) {S}_Y_lm(p) b_{l S} t_lm

        * s must be >= 0.
        * wsmap = healpix map of scan strategy at desired resolution w(p,s)
        * blm   = complex array with dimensions (lmax+1, mmax+1)
        * alm   = complex array with dimensions (0:(lmax+1)*(lmax+2)/2).
                  indexed such that a_{lm} is at position (l*(l+1)/2 + m)
    """
    assert(s >= 0)

    nside     = hp.npix2nside( len(wsmap[:]) )

    alm_lmax  = int((-3 + np.sqrt(1+8*len(alm)))/2)
    assert(alm_lmax == (-3. + np.sqrt(1.+8.*len(alm)))/2.)
    
    blm_lmax  = np.shape(blm[:,:])[0] - 1
    blm_mmax  = np.shape(blm[:,:])[1] - 1
    assert(blm_mmax >= s)

    lmin = s
    lmax = min(blm_lmax, alm_lmax)

    # produce v_{lm} = b_{l S} T_{l m}
    vlm = np.zeros( (lmax+1)**2, dtype=np.complex )
    for l in xrange(lmin, lmax+1):
        vlm[(l*l+l):((l+1)*(l+1))] = alm[l*(l+1)/2:(l+1)*(l+2)/2] * blm[l,s] # +ve m.
        vlm[(l*l):(l*l+l)]         = (np.array([(-1)**m for m in xrange(1,l+1)] ) *
                                      np.conj(alm[(1+l*(l+1)/2):(l+1)*(l+2)/2]) * blm[l,s])[::-1] # -ve m

    cmap = healpix.vlm2map(nside, vlm, s)
    del vlm

    if s > 0: # contribution from negative s.
        cmap *= 2.0

    cmap *= np.conj(wsmap[:])
    
    return cmap.real

# --- misc utility routines ---

def blm_gauss(fwhm, lmax):
    c = (fwhm * np.pi/180./60.)**2 / (16.*np.log(2.0));

    ls  = np.arange(0, lmax+1)
    ret = np.zeros( (lmax+1, 1) )
    ret[:,0] = np.exp(-c*ls*(ls+1.))
    
    return ret

def blm2val(blm, theta, phi):
    # get the value of the beam at the given theta, phi.
    lmax = blm.shape[0] - 1
    mmax = blm.shape[1] - 1

    npts = len(theta)
    zvec = np.cos(theta)

    lfac = np.array( [(2.*l+1.)/(4.*np.pi) for l in xrange(0, lmax+1)] )
    ret = maths._mathsc.wignerd_cf_from_cl( 0, 0, 1, npts, lmax, zvec, blm[:,0].real * lfac )
    for m in xrange(1, mmax+1):
        expp = np.cos(m*phi) + np.sin(m*phi)*1.j
        ret += 2.0 * ( (maths._mathsc.wignerd_cf_from_cl( 0, -m, 1, npts, lmax, zvec, blm[0:(lmax+1),m].real * lfac ) +
                        maths._mathsc.wignerd_cf_from_cl( 0, -m, 1, npts, lmax, zvec, blm[0:(lmax+1),m].imag * lfac ) * 1.j ) * expp ).real

    return ret

def ruzeblm(lmax, freq, lc, rms):
    import gauss_hermite
    assert(lmax >= 0)

    glq = maths.gauss_legendre_quadrature(lmax+1)
    blm = np.zeros( (lmax+1, 1), dtype=np.complex )

    val, gain = gauss_hermite.ruzeval( np.arccos(glq.zvec) * 180.*60./np.pi, freq, lc, rms )
    blm[:,0] = glq.cl_from_cf( lmax, 0, 0, val )

    lfac = np.array( [(2.*l+1.)/(4.*np.pi) for l in xrange(0, lmax+1)] )
    blm[:,0] *= (2.*np.pi)

    return blm

def tpg2blm( lmax, mmax, thts, phis, dthts, dphis, tpg ):
    # harmonic transform of a beam map tpg[itheta, iphi] which
    # is defined on a grid of thts, phis in radians.
    #
    # effectively performs:
    #
    # b[l,m] = \sum_{ itht, (tht, dtht) in enumerate( zip(thts, dthts) ) }
    #          \sum_{ iphi, (phi, dphi) in enumerate( zip(phis, dphis) ) }
    #             Y^{*}_{lm}(tht, phi) tpg[itht, iphi] sin(tht) * dtht * dphi

    ntht = len(thts)
    nphi = len(phis)

    assert( ntht == len(dthts) )
    assert( nphi == len(dphis) )
    assert( np.shape(tpg) == ( ntht, nphi ) )

    costhts = np.cos(thts)
    sinthts = np.sin(thts)

    blm = np.zeros( (lmax+1, mmax+1), dtype=np.complex )
    for m in xrange(0, mmax+1):
        parr = ( np.cos( m * phis ) - 1.j * np.sin( m * phis ) ) * dphis

        tarr = np.dot( tpg, parr )
        warr = dthts * sinthts

        blm[:,m] = ( maths._mathsc.wignerd_cl_from_cf( 0, -m, 1, ntht, lmax, costhts, warr, tarr.real ) +
                     maths._mathsc.wignerd_cl_from_cf( 0, -m, 1, ntht, lmax, costhts, warr, tarr.imag ) * 1.j )

    return blm

def tck2blm( lmax, mmax, nthts, nphis, tck, normalize=False, thtmax=None, interpscale=None):
    import scipy.interpolate

    print 'tck2blm: thtmax = ', thtmax * 180.*60./np.pi, 'arcmin'

    dtht_interp = thtmax / (nthts-1.) * np.ones( nthts)
    thts_interp = np.arange(0, nthts) * dtht_interp

    dphi_interp = 2.*np.pi / nphis * np.ones(nphis)
    phis_interp = np.arange(0, nphis) * dphi_interp

    bmap_interp = np.zeros( (nthts, nphis) )
    for i in xrange(0, nthts):
        for j in xrange(0, nphis):
            bmap_interp[i,j] = scipy.interpolate.bisplev( thts_interp[i] * np.sin(phis_interp[j]) * 180.*60./np.pi,
                                                          thts_interp[i] * np.cos(phis_interp[j]) * 180.*60./np.pi, tck )

    if interpscale != None: bmap_interp = interpscale(bmap_interp)

    ret = tpg2blm( lmax, mmax, thts_interp, phis_interp, dtht_interp, dphi_interp, bmap_interp )

    if normalize == True:
        ret[:] /= ret[0,0]

    for m in xrange(1, mmax+1):
        ret[:,m] *= (np.cos(m*np.pi*0.5) - np.sin(m*np.pi*0.5)*1.j)
        # FIXME: sign/rotation to be resolved! current set of parameters
        #        gives agreement w/ FICSBell for test_arrowbeam (see also
        #        ghb.ghb2blm).

    return ret
