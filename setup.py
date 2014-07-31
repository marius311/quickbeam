#!/usr/bin/env python

healpix_incdir = "/Users/dhanson/Desktop/lib/Healpix_2.15a/src/f90/mod"
healpix_libdir = "/Users/dhanson/Desktop/lib/Healpix_2.15a/lib"

cfitsio_libdir = "/Users/dhanson/Desktop/lib/cfitsio3250"

import glob

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('quickbeam',parent_package,top_path)
    config.add_extension('_healpix', ['quickbeam/_healpix.pyf', 'quickbeam/_healpix.f90'],
                         include_dirs=[healpix_incdir],
                         library_dirs=[healpix_libdir, cfitsio_libdir],
                         libraries=['healpix', 'cfitsio', 'gomp'],
                         extra_compile_args=['-fopenmp'])
    config.add_extension('_mathsc', ['quickbeam/_mathsc.pyf', 'quickbeam/_mathsc.c'])
    config.add_extension('_ghb', ['quickbeam/_ghb.f'])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='quickbeam',
          packages=['quickbeam'],
          configuration=configuration)
