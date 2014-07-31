# maths.py

import numpy as np
import _mathsc

class gauss_legendre_quadrature:
    # Gauss-Legendre and Wigner-d matrix
    # code from libkms_ist, by Kendrick Smith.
    """
    self.npoints = number of points in quadrature
    self.zvec = list of points in [-1,1]
    self.wvec = integration weights for each point
    """

    def __init__(self, npoints):
        self.npoints = npoints
        self.zvec, self.wvec = _mathsc.init_gauss_legendre_quadrature(npoints)

    def cf_from_cl(self, s1, s2, cl):
        "This computes (sum_l C_l d^l_{ss'}), not (sum_l (2l+1)/(4pi) C_l d^l_{ss'})"

        lmax = len(cl)-1

        if np.iscomplexobj(cl):
            #FIXME: convert to 1 cf_from_cl call for potential 2x speed boost.
            return (_mathsc.wignerd_cf_from_cl( s1, s2, 1, self.npoints, lmax, self.zvec, cl.real ) +
                    _mathsc.wignerd_cf_from_cl( s1, s2, 1, self.npoints, lmax, self.zvec, cl.imag ) * 1.j) 
        else:
            return (_mathsc.wignerd_cf_from_cl( s1, s2, 1, self.npoints, lmax, self.zvec, cl ))

    def cl_from_cf(self, lmax, s1, s2, cf):
        "This computes (int f d^l_{ss'}), not 2pi (int f d^l_{ss'})"

        if np.iscomplexobj(cf):
            #FIXME: convert to 1 cl_from_cf call for potential 2x speed boost.
            return (_mathsc.wignerd_cl_from_cf( s1, s2, 1, self.npoints, lmax, self.zvec, self.wvec, cf.real ) +
                    _mathsc.wignerd_cl_from_cf( s1, s2, 1, self.npoints, lmax, self.zvec, self.wvec, cf.imag ) * 1.j)
        else:
            return (_mathsc.wignerd_cl_from_cf( s1, s2, 1, self.npoints, lmax, self.zvec, self.wvec, cf ))
