#!/usr/bin/env python

# sanity test for "bpcl_term" at s=0,
# where the "scan strategy" object is
# just a mask.

import numpy  as np
import pylab  as pl
import healpy as hp

import quickbeam as qb

lmax  = 64
nside = 64
nsims = 100

npix  = 12*nside**2

camb_ls = np.arange(2,lmax+1)
camb_cl = np.loadtxt("inputs/bestfit_scalCls.dat")[:,1][0:lmax-1]

clls = np.arange(0, lmax+1)
cltt = np.concatenate( [ [0, 0], camb_cl * 2.*np.pi / camb_ls / (camb_ls + 1) ] )

mask = np.ones(npix, dtype=np.complex)
mask[ np.where( np.abs(hp.pix2vec( nside, np.arange(0,npix) )[2]) < np.sin(30.*np.pi/180.) ) ] = 0.0

mvlm = qb.healpix.map2vlm( lmax, mask, 0)

fsky = np.sum(mask[:])/npix

cl_avg = np.zeros(lmax+1)
for i in xrange(0,nsims):
    print 'running sim =', i
    tvlm = np.zeros( (lmax+1)**2, dtype=np.complex )
    for l in xrange(0, lmax+1):
        tvlm[(l*l+l)] = np.random.standard_normal() * np.sqrt(cltt[l])
        tvlm[(l*l+l+1):(l+1)**2] = (np.random.standard_normal(l) + np.random.standard_normal(l) * 1.0j) * np.sqrt(cltt[l]) / np.sqrt(2.)
        tvlm[(l*l):(l*l+l)]      = (np.array([(-1)**m for m in xrange(1,l+1)] ) *
                                    np.conj(tvlm[(l*l+l+1):(l+1)**2]))[::-1] # -ve m
    
    tmap = qb.healpix.vlm2map(nside, tvlm, 0)
    tmap *= mask

    tvlm = qb.healpix.map2vlm(lmax, tmap, 0)
    
    cl_avg += qb.spectra.vlm_cl( tvlm )
cl_avg /= nsims

blm  = np.ones( (lmax+1,1) )
bpcl = np.real( qb.bpcl_term(lmax, 0, 0, qb.spectra.vlm_cl(mvlm), blm, blm, cltt) )

pl.loglog( cltt, color='k' )
pl.loglog( cltt * fsky, color='k', linestyle='--' )
pl.loglog( cl_avg.real, label='sim', color='g')
pl.loglog( bpcl, label='theory', color='r')

pl.legend(loc='upper right')
pl.setp( pl.gca().get_legend().get_frame(), visible=False )

pl.show()
