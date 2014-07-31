import numpy as np
import quickbeam as qb
    
def export_dpc_balm_to_fits(objprefix, fitsfname):
    import pyfits
    import piolib

    ix = piolib.ReadVECTObject(objprefix + '_idx', "PIOINT", "")
    re = piolib.ReadVECTObject(objprefix + '_re', "PIOFLOAT", "")
    im = piolib.ReadVECTObject(objprefix + '_im', "PIOFLOAT", "")

    l = np.floor( np.sqrt(ix-1) )
    m = ix - l*l - l - 1

    lmax = np.int(np.max(l))
    mmax = np.int(np.max(m))

    hdu = pyfits.new_table( [pyfits.Column(name='INDEX', format=r'1J', array=ix),
                             pyfits.Column(name='REAL',  format=r'1D', array=re),
                             pyfits.Column(name='IMAG',  format=r'1D', array=im)] )
    
    hdu.header.update('MAX-LPOL', lmax, "Maximum L multipole order")
    hdu.header.update('MAX-MPOL', mmax, "Maximum M multipole degree")
    hdu.header.update('SOURCE', objprefix)
    hdu.name = 'ALM'

    hdu.writeto(fitsfname)

def read_fits_blm(tfname, lmax=None, mmax=None, isbalm=False, colnames=['INDEX','REAL','IMAG']):
    import pyfits
    hdu  = pyfits.open(tfname)[1]

    ix = hdu.data.field(colnames[0])
    re = hdu.data.field(colnames[1])
    im = hdu.data.field(colnames[2])

    ls = np.array( np.floor( np.sqrt(ix - 1) ), dtype=np.int )
    ms = ix - ls*ls - ls - 1

    if lmax == None: lmax = np.max(ls)
    if mmax == None: mmax = np.max(ms)
    
    idxs = (ls <= lmax) * (ms <= mmax)

    ret = np.zeros( (lmax+1, mmax+1), dtype=np.complex)
    ret[ls[idxs], ms[idxs]] = re[idxs] + im[idxs]*1.j

    if isbalm==True: #file is balm, so renormalize and scale.        
        tfacb0 = ret[0,0]
        for l in xrange(0, lmax+1):
            ret[l,:] *= 1.0 / (tfacb0 * np.sqrt((2.*l+1.)))

        for m in xrange(1, mmax+1):
            ret[:,m] *= (np.cos(m*np.pi*0.5) - np.sin(m*np.pi*0.5)*1.j)
        # FIXME: sign/rotation to be resolved! current set of parameters
        #        gives agreement w/ FICSBell for test_arrowbeam (see also
        #        ghb.ghb2blm).

    return ret

def write_fits_blm(fitsfname, blm, colnames=['INDEX','REAL','IMAG']):
    import pyfits

    lmax,mmax = np.shape(blm)
    lmax-=1; mmax-=1

    n_alms = 0
    for l in np.arange(0,lmax+1,1):
        for m in np.arange(0,mmax+1,1):
            if (m<=l):
                n_alms += 1

    idxs = []; re = []; im = [];
    for l in np.arange(0,lmax+1,1):
        for m in np.arange(0,min(lmax, mmax)+1,1):
            idxs.append( l*l + l + m + 1 )
            re.append( blm[l,m].real )
            im.append( blm[l,m].imag )

    hdu = pyfits.new_table( [pyfits.Column(name=colnames[0],  unit='number',format='1J', array=idxs),
                             pyfits.Column(name=colnames[1],  unit='[arbitrary]',format='1D', array=re),
                             pyfits.Column(name=colnames[2],  unit='[arbitrary]',format='1D', array=im)] )

    hdu.header.update('MAX-LPOL', lmax, "Maximum L multipole order")
    hdu.header.update('MAX-MPOL', mmax, "Maximum M multipole degree")
    hdu.header.update('SOURCE', 'quickbeam')
    hdu.name = 'ALM'

    hdu.writeto(fitsfname)

def read_ghb_text(coeffs, params, det="", nmax=9):
    import ghb
    ret = ghb.ghb()
    ret.read_from_textfile(coeffs, params, det, nmax)
    return ret
