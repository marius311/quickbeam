import numpy as np

import maths

def vlm_cl_cross(vlm1, vlm2, lmax=None):
    vlm1_lmax = int(np.sqrt(len(vlm1))-1)
    vlm2_lmax = int(np.sqrt(len(vlm2))-1)
    
    assert(vlm1_lmax == np.sqrt(len(vlm1))-1)
    assert(vlm2_lmax == np.sqrt(len(vlm2))-1)
    
    if lmax == None:
        lmax = min( vlm1_lmax, vlm2_lmax)
    assert(lmax <= min(vlm1_lmax, vlm2_lmax))

    ret = np.zeros(lmax+1, dtype=np.complex)
    for l in xrange(0, lmax+1):
        ret[l] = np.sum( vlm1[(l*l):(l+1)*(l+1)] * np.conj( vlm2[(l*l):(l+1)*(l+1)]) ) / (2.*l+1.)
    return ret

def vlm_cl(vlm, lmax=None):
    return vlm_cl_cross(vlm, vlm, lmax)

def vlm_cl_cross_gc(vlm1, vlm2, lmax=None):
    vlm1_lmax = int(np.sqrt(len(vlm1))-1)
    vlm2_lmax = int(np.sqrt(len(vlm2))-1)

    assert(vlm1_lmax == np.sqrt(len(vlm1))-1)
    assert(vlm2_lmax == np.sqrt(len(vlm2))-1)

    if lmax == None:
        lmax = min( vlm1_lmax, vlm2_lmax)
    assert(lmax <= min( vlm1_lmax, vlm2_lmax))

    retg = np.zeros(lmax+1)
    retc = np.zeros(lmax+1)
    for l in xrange(0, lmax+1):
        tmon = np.sum( vlm1[(l*l):(l+1)*(l+1)] * np.conj( vlm2[(l*l):(l+1)*(l+1)]) )  / (2.*l+1.) * 0.5
        tdif = np.sum( vlm1[(l*l):(l+1)*(l+1)] * vlm2[(l*l):(l+1)*(l+1)][::-1] * np.array([(-1)**m for m in xrange(-l,l+1)]) ).real / (2.*l+1.) * 0.5

        retg[l] = tmon + tdif
        retc[l] = tmon - tdif
    return retg, retc

def vlm_cl_gc(vlm, lmax=None):
    return vlm_cl_cross_gc(vlm, vlm, lmax)

def blm_cl_cross(blm1, blm2, lmax=None, mmax=None, mwt=None):
    assert( blm1.shape == blm2.shape )
    
    if lmax == None: lmax = (blm1.shape[0]-1)
    if mmax == None: mmax = (blm1.shape[1]-1)

    if mwt == None: mwt = np.ones( (mmax+1) )
    assert(len(mwt) > mmax)

    assert(mmax >= 0)
    assert(lmax < blm1.shape[0])
    assert(mmax < blm1.shape[1])
    assert(mwt[0].imag == 0.)

    ret = blm1[0:(lmax+1), 0].real * blm2[0:(lmax+1), 0].real * mwt[0]
    for m in xrange(1, mmax+1):
        ret += 2. * (blm1[0:(lmax+1), m] * np.conj(blm2[0:(lmax+1), m]) * mwt[m]).real

    return ret

def blm_cl_m_cross(blm1, blm2, m, lmax=None):
    assert( blm1.shape == blm2.shape )
    if lmax == None: lmax = (blm1.shape[0]-1)

    assert(np.all(blm1[0:(lmax+1),0].imag == 0.))
    assert(np.all(blm2[0:(lmax+1),0].imag == 0.))

    if m == 0:
        return blm1[0:(lmax+1),0].real * blm2[0:(lmax+1),0].real
    else:
        assert( m >= 0 )
        assert( m <= (blm1.shape[1]-1) )

        return 2. * (blm1[0:(lmax+1),m] * np.conj(blm2[0:(lmax+1),m])).real

def blm_cl(blm, lmax=None, mmax=None, mwt=None):
    return blm_cl_cross(blm, blm, lmax=lmax, mmax=mmax, mwt=mwt)

def blm_cl_m(blm, m, lmax=None):
    return blm_cl_m_cross(blm, blm, m, lmax=lmax)

def calc_cl_lensed(lmax, cl, dcl_g, dcl_c=None, do_r=True):
    """ calculates the first-order lensed temperature cl,
    for a lensing deflection field with (gradient,curl)
    power spectra given by (dcl_g, dcl_c).
        * by default the curl spectrum is zero
        * set do_r=False to neglect the term originating
        from the 2nd derivatve (eq. 4.13 of arxiv:0601594).
    """

    if dcl_c == None: dcl_c = np.zeros(len(dcl_g))
    assert( len(dcl_g) == len(dcl_c) )

    lmax_cl = len(cl)-1
    lmax_dd = len(dcl_g)-1

    dcl_p = dcl_g + dcl_c
    dcl_m = dcl_g - dcl_c

    ngl = (lmax + lmax_cl + lmax_dd)/2+1
    glq = maths.gauss_legendre_quadrature( int(ngl) )

    ls     = np.arange(0, np.max( [lmax_cl, lmax_dd, lmax] )+1)
    llpo   = ls*(ls+1.)
    twolpo = 2.*ls + 1.

    cf_p_cl = glq.cf_from_cl(  1,  1,  cl * twolpo[0:(lmax_cl+1)] * llpo[0:(lmax_cl+1)] )
    cf_p_dd = glq.cf_from_cl( -1, -1,  dcl_p * twolpo[0:(lmax_dd+1)] )

    cf_m_cl = glq.cf_from_cl(  1, -1,  cl * twolpo[0:(lmax_cl+1)] * llpo[0:(lmax_cl+1)] )
    cf_m_dd = glq.cf_from_cl( -1,  1,  dcl_m * twolpo[0:(lmax_dd+1)] )

    ret  = glq.cl_from_cf(lmax, 0, 0, cf_p_cl * cf_p_dd + cf_m_cl * cf_m_dd)
    ret *= 1./(16.*np.pi)

    ret += cl
    if do_r:
        rd   = np.sum( twolpo[0:(lmax_dd+1)] * dcl_p ) / (8.*np.pi)
        ret -= cl * llpo[0:(lmax+1)] * rd

    return ret
