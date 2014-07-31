#!/usr/bin/env python

# compare the beam profile from
# ghb coefficients + ghb2blm to
# the analytical expectation for
# an elliptical Gaussian.

import numpy as np
import pylab as pl

import quickbeam as qb

nmax            = 5
ghb             = qb.ghb.ghb(nmax)

lmax            = 4000
ghb.ellipticity = 1.2
ghb.fwhm        = 10.
ghb.coeffs[0]   = 1.

# --

fwhm_coscan  = ghb.fwhm * np.sqrt(ghb.ellipticity)
fwhm_cxscan  = ghb.fwhm / np.sqrt(ghb.ellipticity)

sigma_coscan = fwhm_coscan / (2*np.sqrt(2*np.log(2)))
sigma_cxscan = fwhm_cxscan / (2*np.sqrt(2*np.log(2)))

thetas = np.linspace(max(sigma_coscan, sigma_cxscan)*3*0.001, max(sigma_coscan, sigma_cxscan)*3,100)

norm = 1.0 / (2*np.pi*sigma_coscan*sigma_cxscan*(np.pi/180./60.)**2)
theory_coscan = np.exp(-0.5*(thetas / sigma_coscan)**2) * norm
theory_cxscan = np.exp(-0.5*(thetas / sigma_cxscan)**2) * norm

blm = qb.ghb.ghb2blm(ghb, lmax)

btp_coscan = qb.blm2val(blm, thetas*np.pi/180./60, 0.0)
btp_cxscan = qb.blm2val(blm, thetas*np.pi/180./60, 0.5*np.pi)

pl.plot(thetas, btp_coscan, color='r')
pl.plot(thetas, theory_coscan, color='g', linestyle='--', lw=2)

pl.plot(thetas, btp_cxscan, color='b')
pl.plot(thetas, theory_cxscan, color='m', linestyle='--', lw=2)

pl.axvline(x=ghb.fwhm/2, color='k', linestyle='--')
pl.axvline(x=fwhm_coscan/2, color='r', linestyle='--')
pl.axvline(x=fwhm_cxscan/2, color='m', linestyle='--')
pl.axhline(y=0.5*norm, color='k', linestyle='--')

pl.xlabel(r'$\theta$ ${\rm arcmin}$')
