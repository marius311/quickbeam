      subroutine ghb2balm(lmax, nmax, ncoeff, nspline, nphi, GHcoeff,  
     & x0, y0, psi, sigmax, sigmay, blm)
c ghb2balm
c computes the harmonic transform of the beam described
c by the input Gauss Hermite parameters and coefficients.
      implicit none

      integer lmax, nmax, ncoeff, nspline, nphi
      double precision GHcoeff(0:(ncoeff-1))
      double precision x0, y0, psi, sigmax, sigmay
      double complex blm(0:lmax,0:nmax)
cf2py integer required,intent(in) :: lmax
cf2py integer required,intent(in) :: nmax
cf2py integer required,intent(in) :: ncoeff
cf2py integer required,intent(in) :: nspline
cf2py integer required,intent(in) :: nphi
cf2py double precision required,intent(in), dimension(ncoeff) :: GHcoeff
cf2py double precision required,intent(in) :: x0
cf2py double precision required,intent(in) :: y0
cf2py double precision required,intent(in) :: psi
cf2py double precision required,intent(in) :: sigmax
cf2py double precision required,intent(in) :: sigmay
cf2py double complex intent(out),dimension(lmax+1,nmax+1) :: blm

      double precision phi_spline(nspline)

      integer il, m, iphi, ispline
      double precision ell,lx,ly

      double precision re_spline(nspline),re2_spline(nspline)
      double precision im_spline(nspline),im2_spline(nspline)
      double precision phi_int(nphi)
      double precision re_int,im_int
      
      double precision beamnorm
      double precision norm(0:nmax)

      double precision pi

      double complex GHbeamFT

      pi=4.d0*atan(1.d0)

c Get Gauss-Hermite normalisation
      call hermite_norm(nmax,norm)
      
      lx=0.d0
      ly=0.d0
      call beamFT(nmax,ncoeff,norm,GHcoeff,x0,y0,psi,sigmax,sigmay,
     &            lx,ly,GHbeamFT)
      beamnorm=abs(GHbeamFT)**2

c Main loop! fill the b_lm. 
      blm = dcmplx(0.d0)

      do il=0,lmax
         ell=dble(il)

c Set up spline for azimuthal variation
      do ispline=1,nspline
         phi_spline(ispline)=-2.d0*pi/10.d0/dble(nmax)+dble(ispline-1)/
     &                dble(nspline-1)*(2.d0*pi+4.d0*pi/10.d0/dble(nmax))
         lx=(ell+0.5d0)*cos(phi_spline(ispline)) ! spherical fudge l+1/2
         ly=(ell+0.5d0)*sin(phi_spline(ispline))
         call beamFT(nmax,ncoeff,norm,GHcoeff,x0,y0,psi,sigmax,sigmay,
     &            lx,ly,GHbeamFT)
         re_spline(ispline)=dble(GHbeamFT)
         im_spline(ispline)=aimag(GHbeamFT)
      end do
      call spline(phi_spline,re_spline,nspline,1.d30,1.d30,re2_spline)
      call spline(phi_spline,im_spline,nspline,1.d30,1.d30,im2_spline)

c Loop on m
      do m=0,min(nmax,il)

c Integrate over phi
      do iphi=1,nphi
         phi_int(iphi)=dble(iphi-1)/dble(nphi)*2.d0*pi
         call splint(phi_spline,re_spline,re2_spline,nspline,
     &               phi_int(iphi),re_int)
         call splint(phi_spline,im_spline,im2_spline,nspline,
     &               phi_int(iphi),im_int)
         blm(il,m) = blm(il,m)+dcmplx(re_int,im_int)*
     &             dcmplx(cos(dble(m)*(phi_int(iphi)+pi/2.d0)),
     &                    -sin(dble(m)*(phi_int(iphi)+pi/2.d0)))
      end do

      blm(il,m)=blm(il,m)/dble(nphi)*2.d0*pi*sqrt((ell+0.5d0)/2.d0/pi)/
     &          (2.d0*pi*sqrt(beamnorm))

      if (m.eq.0) blm(il,m)=dcmplx(real(blm(il,m)))

      end do ! end loop on m

      end do ! end loop on l

      end 

c-----*-------------------------------------------------------------

      subroutine beamFT(nmax,ncoeff,norm,GHcoeff,x0,y0,psi,sigmax,
     &                  sigmay,lx,ly,GHbeamFT)
      implicit none
      integer nmax
      integer ncoeff
      double precision norm(0:nmax)
      double precision GHcoeff(0:ncoeff-1)
      double precision x0,y0
      double precision psi
      double precision sigmax,sigmay
      double precision lx,ly

      double complex GHbeamFT

      integer n1,n2,nindx

      double precision Hn1(0:nmax),Hn2(0:nmax)
      double precision lpx,lpy
      double precision ldotvx0
      double precision cpsi,spsi
      double precision pi,arcmintorad

      pi=4.d0*atan(1.d0)
      arcmintorad=pi/60.d0/180.d0

      cpsi=cos(psi)
      spsi=sin(psi)
      ldotvx0=(lx*x0+ly*y0)*arcmintorad ! \vec{l} \cdot \vec{x}_0
      
      lpx=(lx*cpsi+ly*spsi)*sigmax*arcmintorad
      lpy=(-lx*spsi+ly*cpsi)*sigmay*arcmintorad

      call hermite(nmax,lpx,Hn1)
      call hermite(nmax,lpy,Hn2)
      GHbeamFT=dcmplx(0.d0,0.d0)
      do n1=0,nmax
         do n2=0,nmax-n1        ! n_1 + n_2 <= nmax
            call cart2indx(n1,n2,nindx)
            GHbeamFT=GHbeamFT+dcmplx(GHcoeff(nindx)*
     &                        Hn1(n1)*Hn2(n2)*norm(n1)*norm(n2))*
     &                        dcmplx(0.d0,-1.d0)**(n1+n2)  
         end do
      end do
      GHbeamFT=GHbeamFT*exp(-0.5d0*(lpx*lpx+lpy*lpy))*sigmax*sigmay*
     &         arcmintorad**2*dcmplx(cos(ldotvx0),-sin(ldotvx0))

      return
      end

c-----*-------------------------------------------------------------

      subroutine hermite(n,x,Hn)
      implicit none
      integer n
      double precision x
      double precision Hn(0:n)

      integer i

      if (n.eq.0) then
         Hn(0)=1.d0
      else if (n.eq.1) then
         Hn(0)=1.d0
         Hn(1)=2.d0*x
      else if (n.gt.1) then
         Hn(0)=1.d0
         Hn(1)=2.d0*x
         do i=2,n
            Hn(i)=2.d0*(x*Hn(i-1)-dble(i-1)*Hn(i-2))
         end do
      end if

      return
      end
      
c-----*-------------------------------------------------------------

      subroutine hermite_norm(n,norm)
      implicit none
      integer n
      double precision norm(0:n)

      integer i
      double precision pi
      
      pi=4.d0*atan(1.d0)

      if (n.eq.0) then
         norm(0)=1.d0
      else
         norm(0)=1.d0
         do i=1,n
            norm(i)=norm(i-1)/sqrt(2.d0*dble(i))
         end do
      end if

      return
      end

c-----*-------------------------------------------------------------

      subroutine cart2indx(n1,n2,n)
      implicit none
      integer n1,n2,n
cf2py integer intent(in) :: n1
cf2py integer intent(in) :: n2
cf2py integer intent(out) :: n

      n=(n1+n2)*(n1+n2+1)/2+n2

      return
      end

c-----*-------------------------------------------------------------

c Numerical Recipes cubic spline interpolation routines

c-----*-------------------------------------------------------------

c Given arrays x(1:n) and y(1:n) containing a tabulated function i.e.,
c y(i)=f[x(i)], with x(1) < x(2) < ... < x(n), and given values yp1 and ypn
c for the first derivative of the interpolating function at points 1 and n,
c respectively, this routine returns an array y2(1:n) of length n which
c contains the second derivatives of the interpolating function at the
c tabulated points x(i). If yp1 and/or ypn are equal to 10**30 or larger,
c the routine is signaled to set the corresponding boundary condition for a
c natural spline, with zero second derivative on the boundary.

      subroutine spline(x,y,n,yp1,ypn,y2)
      implicit none
      integer n,nmax
      double precision yp1,ypn,x(n),y(n),y2(n)
      parameter(nmax=5000)

      integer i,k
      double precision p,qn,sig,un,u(nmax)

c Lower bondary condition to "natural" or specified value

      if (yp1.gt..99d30) then
         y2(1)=0.d0
         u(1)=0.d0
      else
         y2(1)=-0.5d0
         u(1)=(3.d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      end if

c Decomposition loop of tridiagonal system

      do i=2,n-1
         sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
         p=sig*y2(i-1)+2.d0
         y2(i)=(sig-1.d0)/p
         u(i)=(6.d0*((y(i+1)-y(i))/(x(i+1)-x(i))-(y(i)-y(i-1))
     &        /(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*u(i-1))/p
      end do

c Upper bondary condition to "natural" or specified value

      if (ypn.gt..99d30) then
         qn=0.d0
         un=0.d0
      else
         qn=0.5d0
         un=(3.d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      end if
      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.d0)

c Back substitution loop of tridiagonal system
      
      do k=n-1,1,-1
         y2(k)=y2(k)*y2(k+1)+u(k)
      end do

      return

      end

c-----*-------------------------------------------------------------

c Given the arrays xa(1:n) and ya(1:n) of length n, which tabulate a function
c (with the xa(i)'s in order), and given the array y2a(1:n) which is the output
c from spline, and given a value of x, this routine returns a cubic-spline
c interpolated value y.

      subroutine splint(xa,ya,y2a,n,x,y)
      implicit none
      integer n
      double precision x,y,xa(n),y2a(n),ya(n)
      
      integer k,khi,klo
      double precision a,b,h
      
      klo=1
      khi=n

c Find place in table by bisection

 1    if (khi-klo.gt.1) then
         k=(khi+klo)/2
         if (xa(k).gt.x) then
            khi=k
         else
            klo=k
         end if
         goto 1
      end if

c khi and klo now bracket the input value x

      h=xa(khi)-xa(klo)
      if (h.eq.0) pause 'Bad xa input in splint'
      
      a=(xa(khi)-x)/h
      b=(x-xa(klo))/h
      y=a*ya(klo)+b*ya(khi)+
     &  ((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**2)/6.d0
      
      return
      
      end

c-----*-------------------------------------------------------------
     
