#!/usr/bin/env python
# export OMP_NUM_THREADS=4

import time
import numpy as np

import quickbeam as qb

nside = 512
lmax  = 2*nside
spin  = 0

tmap = np.ones(12*nside**2, dtype=np.complex)

tstart = time.time()
tvlm = qb.healpix.map2vlm(lmax, tmap, spin)
dt_map2vlm = time.time() - tstart

print 'map2vlm elapsed: %.5fs' % dt_map2vlm

tstart = time.time()
ret = qb.healpix.vlm2map(nside, tvlm, spin)
dt_vlm2map = time.time() - tstart

print 'vlm2map elapsed: %.5fs' % dt_vlm2map


