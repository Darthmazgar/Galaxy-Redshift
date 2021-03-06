c 
      PROGRAM  ZPHOT
c   1 option : zphot 
c                    -c zphot.para       : config file 
c                      
c     last modif 09/11/00
c     Measurement of Phot. redshift
c     Prelimary version with no spectra extraction 
c     Last modif : 1 : add context in input catalg specifying the 
c                      filters used for each object
c                  2 : if prior used : choice of the band 
c                  3 : correction of NaN values if all bands are UPPER_LIMITS
c                  4 : Add Distance Modulus for applying Abs-Mag  
c                  5 : Add Option on command line
c                  6 : Read more than 1 library 
c                  7 : Add fast mode
c                  8 : Add Interpolation of best z 
c                  9 : Add Zproba in various z intervalles
c                  10: Add zform per type (only for synthetical libraries)
c                  11: Add multiple extinction law
      implicit none
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c  VARIABLES 
      integer*4 chisize,nbf,inlib,wmax,zadapt,maxsize
      INCLUDE 'out_unit.decl'
      INCLUDE 'dim_filt.decl'
      INCLUDE 'dim_zchi.decl'
      INCLUDE 'dim_lib.decl'
      INCLUDE 'dim_wave.decl'      
c      parameter (wmax=8000)
      parameter (zadapt=30000)
      parameter (maxsize=110000)
c
c  libraries
      integer*4  numlib,colibmax,nebv,nzf,imagm,nmodg
      integer*4  modext(20)
      integer*4  modlib(inlib),extlib(inlib),reclib(inlib),typlib(inlib)
      integer*4  fobs(chisize,nbf,nbf),fobs4(chisize,nbf,nbf)
      real*8     minkcol(chisize,nbf,nbf)
      real*8     ebvlib(inlib),dmlib(inlib)
      real*8     zlib(inlib),agelib(inlib),ageuniv(inlib)
      real*8     maglib(nbf,inlib),klib(nbf,inlib),emlib(nbf,inlib)
      real*8     zflib(inlib),maglibf(nbf,inlib),ldustlib(inlib)
      real*8     zstep,zmax,dz,h0,om0,l0
      real*8     zf(500),ebv(500)
      
c  chi2 analysis for parameters :extract min , max for Ldust,Age, M, SFR, SSFR within z+/-68%
      real*8     zchipara(12,inlib),parainf(6),parasup(6),paramed(6) 
c  length for phys parameters 50,110000 may be changed here and read_lib prog. 
      real*8     physpara(50,maxsize)
      character*4096 valf(nbf),colib(5),extlaw(10),fileop(81)
      character*4096 magtyp,gallib,qsolib,starlib,emlines
c  best physparameter from Chi_2
      integer*4  reclphysb
      real*8     physpbest(50)
c  prior Lir
      character*4096  lirprior 
      real*8          lirinp,dlirinp,lirpweight
c      
c  for emission lines or output spectra
      integer*4 iopa(81),iext(10),nextlaw,iextinc
      real*8    extic(10,2,wmax),extinc(2,wmax)
      real*8    opal(81,wmax),opat(81,wmax)
c
c  FIR Library 
      character*4096 libext(5),libfir(5),substar,fir_frsc  
      integer*4      nlibext,nfir
      integer*4      liblfir(inlib),modlfir(inlib),reclfir(inlib)
      real*8         zlfir(inlib),lumlfir(inlib),maglfir(nbf,inlib)
      real*8         klfir(nbf,inlib),kzobs(nbf),kccor(nbf)
      real*8         dmagir(5),lumlirb(5),lmini,lmaxi
      real*8         lumtemp(inlib), dmirtemp(inlib),ltemp(inlib)
      integer*4      kbestir(5),sens,ntempir,kmin
      real*8         akir,kcfirb(nbf),magfirb(nbf),fir_lmin
c  FIR Chi 2
      integer*4      nbufir,nbsfir,busfir(nbf),bscfir(nbf),kbest
      integer*4      nf_fir,modfirb,reclfirb,libfirb,lirmax 
      integer*4      irecfirz0,irecfirb
      real*8         fir_cont,fir_scale
      real*8         chi2_fir,dmir,chirmin,dmfirb,zfirb,lumfirb
      real*8         abc(nbf),sabc(nbf),lirmed,lirinf,lirsup
      real*8         lumir(400),chilir(400),dlir,xmin,xmax
      real*8         cont_fir,dmag,absfir(nbf),absfiro(nbf)
c  PHYS Library 
      character*4096 libphys,libphys_sed
      integer*4      nlibphys,reclpbest,phys_max
      real*8         libppara(50,maxsize),phys_cont,phys_scale,dmpbest 
c      integer*4      nbuphys,nbsphys,busphys(nbf),bscphys(nbf)  
      real*8         zrecp(500),chipbest
      real*8         fluxphys(nbf)
      real*8         kcorphys(nbf),magphys0(nbf)
      integer*4      zrecpi(500),zrecps(500),nzrecp,nppara(50)
      real*8        ppbest(50),ppmed(50),ppinf(50),ppsup(50)
c
c  global       
      integer*4 imag,imax,imin,nf,i,j,k,nobj,nobjm
      integer*4 nchi,spec,model,lnblnk
      integer*4 bp,bused(nbf),bdincl,babs
      integer*4 extil,recl,extilb(chisize),reclb(chisize) 
      integer*4 reclmin(3),extilmin(3)
      integer*4 recb(chisize),recb0(chisize),imasb(chisize),recmin(3)
      integer*4 chimax,chitrans,pass
      integer*4 nlib,imasmin(3),ndz,nb(500)
      integer*4 nsp,nspmax,nspmaxg,nspmaxs,nspmaxq
      integer*4 nsp1,nsp2,nspfir,nspphys
      integer*4 irecz0,buscal(nbf)
      integer*4 nerr,npdz
      integer*4 nmeas(5)
      integer*4 paravi(500),test,nbused,liblength,nmod
      integer*4 iwpara,iwout,npara(500)
      integer*4 nbul,nbus
      integer*4 cat_fmt,rowmin,rowmax
c  CONTEXT
      integer*4 conti
      real*8    cont,gbcont,bdscal
      real*8    contforb,new_cont
c
      real*8    zfb(chisize),zfmin(3),zfmod
      real*8    fsp(wmax),fgal(5,wmax),wgal(5,wmax),fq(wmax),wq(wmax)
      real*8    wsp(wmax),fst(wmax),wst(wmax),magabsl(2)
      real*8    lb,emspec(2,wmax),y1(wmax),y2(wmax),lsamp(wmax)
      integer*4 nem,lsmax
      real*8    extb(chisize),extmin(3),exti,mod_dist2,z0
      real*8    mag_abs(3),mag_absb(chisize),dist_mod(chisize)
      real*8    mod_distb(3),mabs(nbf),mabsq(nbf),kapq(nbf),kap(nbf)
      real*8    abs_mag,lbdmaxwr,lbdminwr
      real*8    ageb(chisize),dmb(chisize),zb(chisize)
      real*8    kcorb(nbf,chisize),ldustb(chisize)
      real*8    zp(nbf),mag(nbf),kcor(nbf),magb(nbf),z,age,zs,ldust
      real*8    ab(nbf),sab(nbf),abo(nbf),sabo(nbf),avmagt,avmago
      real*8    aborig(nbf),saborig(nbf)
      real*8    zmin(3),dm,agemin(3),dmmin(3),dmcor,zintb,zbest,ztemp
      real*8    chi2,chimin(3),chi(2,chisize),chibest,ldustmin(3)
      real*8    xp(chisize),yp(chisize)
      real*8    abcor(nbf),flmoy(nbf),flwidth(nbf)
      real*8    maxlz(chisize),dzml,summl,mlarea,pdz(500)
      real*8    zinf,zsup,zmin_gal,zmax_gal,ebvmin,ebvmax
      real*8    int_pdz(500),dzpdz(500)
      real*8    funz,funz0,lmasi,lmass,fac_err
      real*8    fcorr(nbf)
c BAYESIAN PHOTO-Z 
      real*8    chibay(chisize),zbay,zmed,zbayi,zbays
      real*8    barea
c PRIORS 
      real*8    pweight,iab,dzp
      real*8    nzprior,nzprior2,nzprior3,nzprior4
c      real*8    nzpriorVVDS5
      real*8    color_rf
      integer*4 bp_B,bp_I
c GLOBALS
      real*8    timy,zform(500),zform2,tused,tzform,tuniv
      real*8    paravr(500),dz_win,min_thres,min_err(500)
      real*8    dchi,z68i,z68s,z90i,z90s,z99i,z99s
      real*8    val1,val2
c      real*8    pb68,pb90,pb99
      real*8    probz(10),zpdzi(10),zpdzs(10)
      real*8    zml68i,zml68s,zml90i,zml90s,zml99i,zml99s
      character*4096 str_inp,str
      character*4096 param
      character*4096 paravc(500),str_ch
      character*4096 zpdir,zpwork,ospec,config,file
      character*4096 filters,cat,outf
      character*4096 catmag,zfix,cattyp
      character*4096 outsp,typm,outchi
      character*4096 outpara,wpara(500),str_out(500)
      character*4096  cr_date,fdate
c  LIBRARY SORTED BY COLORS 
      character*4096 sel,fastmod,zintp
      integer*4 numcol,f_index(nbf,inlib),reclist(inlib)
      integer*4 numrec,redoing
      real*8    sigcol,fsort(nbf,inlib),magm(nbf)
c  ABSOLUTE MAGNITUDES 
      integer*4  method(nbf),nmeth
      integer*4  bapp(nbf),recmin0(3),goodfilter(nbf),index
      real*8     magm0(nbf),minkcolor(nbf),macont
      integer*4  magabscont(nbf),mbused(nbf,nbf)
      integer*4  zmfilt(500),nzmax,zband
      real*8     zmlim(500)
      real*8     zvmax(500),diff_mag,zmaxlib
c  ABS-MAG with method=4
      integer*4  l,m,nbBinZ,bappOp(500)
      real*8     zbmin(500),zbmax(500)
c  OUTPUT FILES for PDZ and ABS filter
      integer*4      nfabs,ufabs,fabs
      integer*4      pdz_fabs(nbf)
c      integer*4      pdz_cont(nbf),pdz_meth(nbf)
      real*8         pdz_mabsz,pdz_mabs(nbf,chisize),pdz_kcor(nbf)
      real*8         pdz_z,pdz_dm,pdz_distm
      character*4096  outpdz,pdz_file,pdz_abs(nbf),pdz_mod,pdz_zph
c
c  AUTO ADAPT   
      character*4096 autoadapt,adapterror
      integer*4 degre,nbshift,mod_ada(zadapt)
      real*8    cont_ada(zadapt),adcont
      integer*4 meth_ada,ngals_ada,admmin,admmax
      integer*4 realise,iteration,fl_auto,fl1,fl2,iter_best
      real*8    x,x2,x3,auto_thresmin,auto_thresmax
      real*8    ab_ada(zadapt,nbf),sab_ada(zadapt,nbf)
      real*8    magm_ada(zadapt,nbf),zs_ada(zadapt)
      real*8    a0(nbf),a0in(nbf),a1(nbf),a1in(nbf),a2(nbf),a2in(nbf)
      real*8    a3(nbf),a3in(nbf)
      real*8    a0best(nbf),a1best(nbf),a2best(nbf),a3best(nbf)
      real*8    min_errbest(nbf)
      real*8    residu,res_best,chiin,chifit
      real*8    adzmin,adzmax,corr,shift(nbf)
      real*8    maglibini(nbf,inlib),maglibfini(nbf,inlib)
c
      common /func_int/  k,ngals_ada,fl1,fl2,fl_auto,meth_ada,imagm
      common /func_tab/  ab_ada,sab_ada,magm_ada,zs_ada
      common /func_tab2/ cont_ada,mod_ada
      common /min_err/   min_err
c
c  EMISSION LINES  
      integer*4  imuv
      real*8     lambf_UV(wmax),repf_UV(wmax)
      real*8     ext_em(7),aext_lb(10,7)
      real*8     em(nbf),em2(nbf),emMin(nbf),mabsuv,nuvr 
c      real*8     abs_magU
      character*4096  addem
      real*8     frac(6),fracMin,dmmi,chiMinEm
      real*8      Lsol
      parameter  (Lsol=3.826e33)
c
c  CHI2 CHECK per band 
      real*8     chi2_fl(nbf),chi2_fl_min(nbf),sum_chi2
      real*8     levelChi2,levelMag
      integer*4  redofit
c
      EXTERNAL      bdincl 
      external      funz,nzprior,nzprior2,nzprior3,nzprior4,timy
c
cccccccccccccccccccccccccccccccccccccccccccccccccccc
c     If you want to give a name to the output screen file
      if(UO.ne.6)then
           open(UO,file='ZPHOT.screen',status='unknown') 
      endif
c
cccccccccccccccccccccccccc
c Environemental  Variable 
      call getenv('LEPHAREDIR',zpdir)
      test=lnblnk(zpdir)
      if (test .eq. 0) then
        write(UO,*) 'WARNING :  variable LEPHAREDIR not defined'
        stop
      endif
      call getenv('LEPHAREWORK',zpwork)
      test=lnblnk(zpwork)
      if (test .eq. 0) then
        write(UO,*) 'WARNING :  variable LEPHAREWORK not defined'
        stop
      endif
ccccccccccccccccccccccccccccccccccccccccccccccccccc
c help on line
      param='zphot'
      call get_help(test)
      if (test .eq. 1) call help(param)
c
cccccccccccccccccccccccccccccccccccccccccccccccc
c Initialisation of Input parameter from config file
       param='-c'
      call get_conf(param,config,test)
      if (test.ne.1) call err_option(param,1)
      call get_path(config)
cccccccc PRIMARY  OPTIONS  ccccccccccccccccc

      param='-CAT_IN'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1) cat = paravc(1)(1:lnblnk(paravc(1)))
      if (test.ne.1) call err_option(param,1)
      call get_path(cat)
c
      param='-INP_TYPE'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1) then
         if (paravc(1)(1:1) .eq. 'F' .OR. 
     >       paravc(1)(1:1) .eq. 'f') typm = "F"
         if (paravc(1)(1:1) .eq. 'M' .OR. 
     >       paravc(1)(1:1) .eq. 'm') typm = "M"
         if (typm(1:1) .ne. 'F' .and.
     >       typm(1:1) .ne. 'M')  call err_option(param,1)
      else
         call err_option(param,1)
      endif
c
      param='-CAT_MAG'  
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1)  then
         if (paravc(1)(1:1) .eq. 'V' .OR. 
     >       paravc(1)(1:1) .eq. 'v') catmag = "VEGA"
         if (paravc(1)(1:1) .eq. 'A' .OR. 
     >       paravc(1)(1:1) .eq. 'a') catmag = "AB"
         if (catmag(1:1) .ne. 'A' .and.
     >       catmag(1:1) .ne. 'V')  call err_option(param,1)
      else
         call err_option(param,1)
      endif
c
      param='-CAT_FMT'  
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1)  then 
         cat_fmt=-1  
         if ( paravc(1)(1:lnblnk(paravc(1)))  .eq. 'MEME' .OR. 
     >        paravc(1)(1:lnblnk(paravc(1)))  .eq. 'meme') cat_fmt=0 
         if ( paravc(1)(1:lnblnk(paravc(1)))  .eq. 'MMEE' .OR.
     >        paravc(1)(1:lnblnk(paravc(1)))  .eq. 'mmee') cat_fmt=1 
         if (cat_fmt.ne.0 .and. cat_fmt.ne.1) call err_option(param,1)
      elseif (test.ne.1) then
         call err_option(param,1)
      endif
c
      param='-CAT_LINES'  
      call geti_option(param,config,2,paravi,test)
      if (test.eq.2)  then
         rowmin=paravi(1)
         rowmax=paravi(2)
         if (rowmin.gt.rowmax) then
            rowmin=-99
            rowmax=-99
         endif
      else
         rowmin=-99
         rowmax=-99
      endif

c      param='-FILTER_FILE'
c      call getc_option(param,config,1,paravc,test)
c      if (test.eq.1)  filters=paravc(1)(1:lnblnk(paravc(1)))
c      if (test.ne.1)  call err_option(param,1)
c
      param='-ZPHOTLIB'  
      call getc_option(param,config,500,paravc,test)
      if (test.ge.1 .and. test.le.3) then
         do i = 1,test
           colib(i)=paravc(i)(1:lnblnk(paravc(i)))
         enddo  
         numlib=test
      elseif (test.gt.3) then
         write(UO,*) 'More than 3 librairies used --> STOP'
         stop
      else
         call err_option(param,1)         
      endif
c
      param='-PARA_OUT'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1) outpara=paravc(1)(1:lnblnk(paravc(1)))
      if (test.ne.1) call err_option(param,1)
      call get_path(outpara)
c
      param='-CAT_OUT'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1) outf=paravc(1)(1:lnblnk(paravc(1)))
      if (test.eq.1) call get_path(outf)
      if (test.ne.1) outf='zphot.out'
c      
ccccccccc  SECONDARY OPTIONS  ccccccccccccccccccc
      param='-CAT_TYPE'  
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1)  then 
         if (paravc(1)(1:1) .eq. 'S' .OR. 
     >       paravc(1)(1:1) .eq. 's') cattyp = "SHORT"
         if (paravc(1)(1:1) .eq. 'L' .OR. 
     >       paravc(1)(1:1) .eq. 'l') cattyp = "LONG"
      else 
        cattyp='SHORT'
      endif
c INPUT ERRORS  TUNING  (Add in Quadrature) 
      param='-ERR_SCALE'
      call getf_option(param,config,500,paravr,test)
      nerr=test
      if (test.ge.1) then
         do i = 1,nerr
           min_err(i) =   paravr(i)
         enddo
      else
         nerr=1
         min_err(1)=-1
      endif   

c SCALING FACTOR 
      param='-ERR_FACTOR'
      call getf_option(param,config,1,paravr,test)
      if (test.eq.1) fac_err=paravr(1)
      if (test.ne.1) fac_err=1.0
c used band for scaling 
      param='-BD_SCALE'
      call geti_option(param,config,1,paravi,test)
      if (test.eq.1)  bdscal=DBLE(paravi(1))
      if (test.ne.1)  bdscal=0.d0
c global context imposed to all objects
      param='-GLB_CONTEXT'
c      call geti_option(param,config,1,paravi,test)
c      if (test.eq.1)  gbcont=DBLE(paravi(1))
      call getf_option(param,config,1,paravr,test)
      if (test.eq.1)  gbcont=paravr(1)
      if (test.ne.1)  gbcont=-1.d0

c rejected bands 
      param='-FORB_CONTEXT'
      call getf_option(param,config,1,paravr,test)
      if (test.eq.1)  contforb=paravr(1)
      if (test.ne.1)  contforb=-1.d0

c reject band with highest contribution to global Chi2
      param='-CHI2_RM_BD'
      call getf_option(param,config,2,paravr,test)
      if (test.eq.2)levelChi2=paravr(1)
      if (test.eq.2)levelMag =paravr(2)
      if (test.ne.2)levelChi2=10000.d0
      if (test.ne.2)levelMag =10000.d0

cccccccccccccc   PRIOR  cccccccccccccccccccccccc   
      param='-MASS_SCALE'  
      call getf_option(param,config,2,paravr,test)
      if (test.eq.2)   lmasi=paravr(1)
      if (test.eq.2)   lmass=paravr(2)
      if (test.ne.2)   lmasi=0.
      if (test.ne.2)   lmass=0.
      param='-MAG_ABS'
      call getf_option(param,config,2,paravr,test)
      if (test.eq.2) magabsl(1)=paravr(1)
      if (test.eq.2) magabsl(2)=paravr(2) 
      if (test.ne.2)   magabsl(1)=0.
      if (test.ne.2)   magabsl(2)=0.
      param='-MAG_REF'
      call geti_option(param,config,1,paravi,test)
      if (test.eq.1)  babs=paravi(1)
      if (test.ne.1)  babs=0
c   prior on N(z) based on z-VVDS: I mag + rest frame colors B-I 
      param='-NZ_PRIOR'  
      call geti_option(param,config,3,paravi,test)
      if (test.eq.3)  bp=paravi(1)
      if (test.eq.3)  bp_B=paravi(2)
      if (test.eq.3)  bp_I=paravi(3)
      if (test.ne.3)  then
        bp=0
        bp_B=0
        bp_I=0
      endif 
c      param='-NZ_PRIOR'  
c      call geti_option(param,config,1,paravi,test)
c      if (test.eq.1)  bp=paravi(1)
c      if (test.ne.1)  bp=0
c
      param='-Z_RANGE'
      call getf_option(param,config,2,paravr,test)
      zmin_gal=-99.99
      zmax_gal= 99.99 
      if (test.eq.2) then 
        if (paravr(1).ge.0 .and. paravr(2).gt.paravr(1)) then 
          zmin_gal=paravr(1)
          zmax_gal=paravr(2)
        endif
      endif
c
      param='-EBV_RANGE'
      call getf_option(param,config,2,paravr,test)
      ebvmin=-99.99
      ebvmax=-99.99
      if (test.eq.2)  then 
        if (paravr(1).ge.0 .and. paravr(2).ge.paravr(1)) then
           ebvmin=paravr(1)
           ebvmax=paravr(2)
        endif           
      endif
c
      param='-ZFORM_MIN'
      call getf_option(param,config,500,paravr,test)
      nmod=test      
      if (test.ge.1) then
         do i = 1, nmod
            zform(i)=paravr(i)
         enddo   
      else
         nmod=1
         zform(1)=0.
      endif   
c
      param='-LIR_PRIOR'
      call getc_option(param,config,1,paravc,test)
      lirprior = "NO"
      if (test.eq.1) then
         if (paravc(1)(1:1) .eq. 'Y' .OR. 
     >       paravc(1)(1:1) .eq. 'y') lirprior="YES"
      endif   

c
cccccccc   FIXED REDSHIFT   cccccccccccccccc
      param='-ZFIX'  
      call getc_option(param,config,1,paravc,test)      
      if (test.eq.1)   then
         if (paravc(1)(1:1) .eq. 'Y' .OR. 
     >       paravc(1)(1:1) .eq. 'y') zfix = "YES"
         if (paravc(1)(1:1) .eq. 'N' .OR. 
     >       paravc(1)(1:1) .eq. 'n') zfix = "NO"
      else
          zfix='NO'
      endif
cccc  REDSHIFT INTERPOLATION BETWEEN Library Z-STEP
      param='-Z_INTERP'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1) then 
         if (paravc(1)(1:1) .eq. 'Y' .OR. 
     >       paravc(1)(1:1) .eq. 'y') zintp = "YES"
         if (paravc(1)(1:1) .eq. 'N' .OR. 
     >       paravc(1)(1:1) .eq. 'n') zintp = "NO"
      else
          zintp="NO"
      endif
c
cccc    SEARCH FOR SECONDARY SOLUTION 
      param='-DZ_WIN'
      call getf_option(param,config,1,paravr,test)
      if (test.eq.1)  dz_win=paravr(1)
      if (test.ne.1)  dz_win=0.25 
c
      param='-MIN_THRES'
      call getf_option(param,config,1,paravr,test)
      if (test.eq.1)  min_thres=paravr(1)
      if (test.ne.1)  min_thres=0.1 
cccc  Integrated PDFz over Z ranges     
      param='-PROB_INTZ'
      call getf_option(param,config,500,paravr,test)
      npdz=test
      if (test.ge.1) then
         if (MOD(npdz,2).eq.0) then
           do i = 1,npdz
             int_pdz(i) =   paravr(i)
           enddo
         else
           npdz=1
           int_pdz(1) = 0.
         endif  
      else
         npdz=1
         int_pdz(1) = 0.
      endif   
ccccc  VMAX
      nzmax=0
      do i = 1,500
         zmfilt(i)=0
         zmlim(i)=0
      enddo
      param='-ZMAX_FILT'
      call geti_option(param,config,500,paravi,test)
      nzmax=test
      if (test.ge.1) then 
       do i = 1,test
         zmfilt(i) = paravi(i)
       enddo
      endif
      param='-ZMAX_MAGLIM'
      call getf_option(param,config,500,paravr,test)
      if (test.ge.1 .and. test.eq.nzmax) then
        do i = 1,test
           zmlim(i)=paravr(i)
        enddo
      else
c        write(U0,*) ' OPTION ZMAX not used '
c      reset the option to OFF   
        nzmax=0
        do i = 1,500
         zmfilt(i)=0
         zmlim(i)=0
        enddo
      endif      
c

c
cccc   OUTPUT INDIVIDUAL SPECTRA  
      param='-SPEC_OUT'
      call getc_option(param,config,1,paravc,test)
      outsp='NO'
      if (test.eq.1)  then 
         if (paravc(1)(1:1) .eq. 'Y' .OR. 
     >       paravc(1)(1:1) .eq. 'y') outsp = "YES"
         if (paravc(1)(1:1) .eq. 'N' .OR. 
     >       paravc(1)(1:1) .eq. 'n') outsp= "NO"
      endif
cccc   OUTPUT ALL CHI2 and PARAMETERS 
      param='-CHI2_OUT'
      call getc_option(param,config,1,paravc,test)
      outchi='NO' 
      if (test.eq.1)  then 
         if (paravc(1)(1:1) .eq. 'Y' .OR. 
     >       paravc(1)(1:1) .eq. 'y') outchi = "YES"
         if (paravc(1)(1:1) .eq. 'N' .OR. 
     >       paravc(1)(1:1) .eq. 'n') outchi= "NO"
      endif

c
cccccccccccccccccccccccccccccccccccccc
c  FIR Libraries 
      param='-FIR_LIB'
      nlibext=0
      call getc_option(param,config,500,paravc,test)
      if (test.ge.1 .and. test.le. 5) then 
        nlibext=test
        do i =1,test
          libext(i)=paravc(i)(1:lnblnk(paravc(i)))
        enddo
        if (libext(1)(1:lnblnk(libext(1))).eq.'NONE' .OR.
     >      libext(1)(1:lnblnk(libext(1))).eq.'NONE') then
          nlibext=0 
        endif
      elseif (test.gt.5) then 
         nlibext=5
         write(UO,*) 'More than 5 librairies defined  '
        do i =1,nlibext
          libext(i)=paravc(i)(1:lnblnk(paravc(i)))
        enddo
      elseif (test.le.0) then
          nlibext=0
          libext(1)='NONE'
      endif
c  given in micron  (um) 
      param='-FIR_LMIN'
      call getf_option(param,config,1,paravr,test)
      if (test.eq.1)  fir_lmin=paravr(1)
      if (test.ne.1)  fir_lmin=7.0 

c
      param='-FIR_CONT'
      call getf_option(param,config,1,paravr,test)
      if (test.eq.1)  fir_cont=paravr(1)
      if (test.ne.1)  then 
        fir_cont=-1.d0
      endif
c
      param='-FIR_SCALE'
      call geti_option(param,config,1,paravi,test)
      if (test.eq.1)  fir_scale=DBLE(paravi(1))
      if (test.ne.1)  then 
        fir_scale=-1.d0
      endif
c     allow for free scaling if fir_frsc=Y [default]  OR NOT if fir_frsc=NO
      param='-FIR_FREESCALE'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1) then 
         if (paravc(1)(1:1) .eq. 'Y' .OR.
     >       paravc(1)(1:1) .eq. 'y') then 
             fir_frsc='YES'
         else
             fir_frsc='NO'
         endif
      else
          fir_frsc='NO'
      endif
c
      param='-FIR_SUBSTELLAR'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1) then 
         substar=paravc(1)(1:lnblnk(paravc(1)))
      else
        substar='NO'
      endif
      if (substar(1:lnblnk(substar)).eq.'YES' .OR. 
     >    substar(1:lnblnk(substar)).eq.'yes') then 
          substar='YES' 
      else
         substar='NO'
      endif     
c
ccccccccccccccccccccccccccccccccccccccccccc
c  PHYSICAL PARAMETERS LIBRARY
      param='-PHYS_LIB'
      nlibphys=0
      call getc_option(param,config,1,paravc,test)
      if (test.ge.1) then 
         nlibphys=1
         libphys=paravc(1)(1:lnblnk(paravc(1)))
         if (libphys(1:lnblnk(libphys)).eq.'NONE' .OR.
     >       libphys(1:lnblnk(libphys)).eq.'none') then
           nlibphys=0 
        endif
      else
         nlibphys=0
         libphys='NONE'
      endif
c
      param='-PHYS_CONT'
      call getf_option(param,config,1,paravr,test)
      if (test.eq.1)  phys_cont=paravr(1)
      if (test.ne.1)  then 
        phys_cont=-1.d0
      endif
c
      param='-PHYS_SCALE'
      call geti_option(param,config,1,paravi,test)
      if (test.eq.1)  phys_scale=DBLE(paravi(1))
      if (test.ne.1)  then 
        phys_scale=-1.d0
      endif
c
      param='-PHYS_NMAX'
      call geti_option(param,config,1,paravi,test)
      if (test.eq.1)  phys_max=paravi(1)
      if (test.ne.1)  then 
        phys_max=100000
      endif

c  
cccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccc  ABSOLUTE MAGNITUDE ESTIMATES  ccccccccccccccccccccccc

      param='-MABS_METHOD'
      call geti_option(param,config,500,paravi,test)
      nmeth=test
      if (nmeth.gt.1 .and. nmeth.le.nbf) then
         do i = 1,nmeth 
            method(i)=paravi(i)
         enddo
      elseif  (nmeth.eq.1) then
         do i = 1,nbf 
            method(i)=paravi(1)
         enddo
      else
         do i = 1,nbf 
            method(i)=0
         enddo
      endif
c
      param='-MABS_CONTEXT'
      call geti_option(param,config,500,paravi,test)
      if (test.gt.1 .and. test.eq.nmeth) then
         do i = 1,nmeth 
            magabscont(i)=paravi(i)
         enddo
      elseif (test.eq.1) then 
        do i = 1,nbf 
            magabscont(i)=paravi(1)
         enddo
      else
         do i = 1,nbf 
            magabscont(i)=-1
         enddo
      endif
c     keyword for MABS_METHOD=2
      param='-MABS_REF'
      call geti_option(param,config,500,paravi,test)
      if (test.gt.1 .and. test.eq.nmeth) then
         do i = 1,nmeth 
            bapp(i)=paravi(i)
         enddo
      elseif (test.eq.1) then
         do i = 1, nbf
            bapp(i)=paravi(1)
         enddo
      else
         do i = 1, nbf
            bapp(i)=i
         enddo
      endif
c     Option of the LF used for MABS_METHOD=4
      param='-MABS_FILT'
      call geti_option(param,config,500,paravi,test)
      call getOpt1Tab(nbBinZ,bappOp,500,test,paravi,param)
c
      param='-MABS_ZBIN'
      call getf_option(param,config,500,paravr,test)
      call getOpt2Tab(nbBinZ,zbmin,zbmax,500,test,paravr,param)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccc    AUTO-ADAPT     METHOD     ccccccccccccccccccccccc
c
      param='-APPLY_SYSSHIFT'
      call getf_option(param,config,500,paravr,test)
      nbshift=test
      if (test.gt.0) then
        do i = 1,nbshift
           shift(i) = paravr(i)
        enddo
      endif
c
      param='-AUTO_ADAPT'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1)  then 
         if (paravc(1)(1:1) .eq. 'Y' .OR. 
     >       paravc(1)(1:1) .eq. 'y') autoadapt = "YES"
         if (paravc(1)(1:1) .eq. 'N' .OR. 
     >       paravc(1)(1:1) .eq. 'n') autoadapt = "NO"
      else
         autoadapt='NO'
      endif
c
      param='-ERROR_ADAPT'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1)  then 
         if (paravc(1)(1:1) .eq. 'Y' .OR. 
     >       paravc(1)(1:1) .eq. 'y') adapterror = "YES"
         if (paravc(1)(1:1) .eq. 'N' .OR. 
     >       paravc(1)(1:1) .eq. 'n') adapterror = "NO"
      else
          adapterror='NO'
      endif
c
      param='-ADAPT_BAND'
      call geti_option(param,config,3,paravi,test)
      if (test.eq.3) then
        fl_auto=paravi(1)
        fl1=paravi(2)
        fl2=paravi(3)
      else
        fl_auto=0
        fl1=0
        fl2=0
      endif
      if(fl_auto.le.0)  autoadapt='NO'
c
      param='-ADAPT_LIM'
      call getf_option(param,config,2,paravr,test)
      if (test.eq.2)then
        auto_thresmin=paravr(1)
        auto_thresmax=paravr(2)
      else
        auto_thresmin=18.0d0
        auto_thresmax=22.d0
      endif 
c
      param='-ADAPT_POLY'
      call geti_option(param,config,1,paravi,test)
      if (test.eq.1 .and.paravi(1).le.4) then 
         degre=paravi(1)
      else
         degre=1
      endif
c
      param='-ADAPT_METH' 
      call geti_option(param,config,1,paravi,test)
      if (test.eq.1)  meth_ada=paravi(1)
      if (test.ne.1)  meth_ada=1
c
      param='-ADAPT_CONTEXT'
      call getf_option(param,config,1,paravr,test)
      if (test.eq.1)  adcont=paravr(1)
      if (test.ne.1)  adcont=-1.d0
c
      param='-ADAPT_ZBIN'
      call getf_option(param,config,2,paravr,test)
      if (test.eq.2) adzmin=paravr(1)
      if (test.eq.2) adzmax=paravr(2) 
      if (test.ne.2) adzmin=0.001
      if (test.ne.2) adzmax=6.
c
      param='-ADAPT_MODBIN'
      call geti_option(param,config,2,paravi,test)
      if (test.eq.2) admmin=paravi(1)
      if (test.eq.2) admmax=paravi(2) 
      if (test.ne.2) admmin=1
      if (test.ne.2) admmax=1000     
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccccccc PDZ OUTPUT cccccccccccccccccccccccccccc
c
      param='-PDZ_OUT'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1)  outpdz=paravc(1)
      if (outpdz(1:4).eq."none")   outpdz='NONE'
      if (test.ne.1)  outpdz='NONE'
      nfabs=0
      if (outpdz(1:4).ne."NONE") then      
c  filter ref  
        param='-PDZ_MABS_FILT'
        call geti_option(param,config,500,paravi,test)
        if (test.ge.1 ) then
           nfabs=test
           do i = 1, nfabs  
              pdz_fabs(i)=paravi(i)
           enddo
        endif
c  opening Mag abs files 
        do i = 1, nfabs           
           write(ospec,'(i2.2)') pdz_fabs(i) 
           pdz_abs(i)=outpdz(1:lnblnk(outpdz))//'.abs'//ospec(1:2)
        enddo
        pdz_file= outpdz(1:lnblnk(outpdz)) // '.pdz'
        pdz_mod = outpdz(1:lnblnk(outpdz)) // '.mod'
        pdz_zph = outpdz(1:lnblnk(outpdz)) // '.zph'
      endif 
c
ccccccccccccc EMISSION LINES  cccccccccccccccccccccccccccc
c
      param='-ADD_EMLINES'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1)  addem=paravc(1)
      if (test.ne.1)  addem='NO'
      if (addem(1:1) .eq. 'Y' .OR. 
     >    addem(1:1) .eq. 'y')  addem = "YES"
c 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccc   FAST MODE By sorting in color space the library 
      param='-FAST_MODE'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1)  then 
         if (paravc(1)(1:1) .eq. 'Y' .OR. 
     >       paravc(1)(1:1) .eq. 'y') fastmod = "YES"
         if (paravc(1)(1:1) .eq. 'N' .OR. 
     >       paravc(1)(1:1) .eq. 'n') fastmod = "NO"
      else
         fastmod='NO'
      endif
c
      param='-COL_NUM'
      call geti_option(param,config,1,paravi,test)
      if (test.eq.1)  numcol=paravi(1)
      if (test.ne.1)  numcol=3
c
      param='-COL_SIGMA'  
      call getf_option(param,config,1,paravr,test)      
      if (test.eq.1)  sigcol=paravr(1)
      if (test.ne.1)  sigcol=3.
c
      param='-COL_SEL'
      call getc_option(param,config,1,paravc,test)
      if (test.eq.1)  sel=paravc(1)
      if (test.ne.1)  sel='OR'
c
cccccccccccccccccccccccccccccccccccccccccccccccccc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  INFO PARAMETERS ON SCREEN 
      write(UO,'(A)')  "#######################################"
      write(UO,'(A)')  "# PHOTOMETRIC REDSHIFT with OPTIONS   #"
      write(UO,'(2A)') "# CAT_IN       : ",cat(1:lnblnk(cat))
      write(UO,'(2A)') "# CAT_OUT      : ",outf(1:lnblnk(outf))
      write(UO,'(A,2(I10,1x))') "# CAT_LINES     : ",rowmin,rowmax
      write(UO,'(2A)') "# PARA_OUT     : ",outpara(1:lnblnk(outpara))
      write(UO,'(2A)') "# INP_TYPE     : ",typm(1:lnblnk(typm))
      write(UO,'(A,2x,I4)') "# CAT_FMT[0:MEME 1:MMEE]: ",cat_fmt
      write(UO,'(2A)') "# CAT_MAG      : ",catmag(1:lnblnk(catmag))
c      write(UO,'(2A)') "# FILTER_FILE  : ",filters(1:lnblnk(filters))
      write(UO,'(7A)') "# ZPHOTLIB     : ",
     >(colib(i)(1:lnblnk(colib(i))),' ',i=1,numlib)
      write(UO,'(2A)') "# ADD_EMLINES  : ",addem(1:lnblnk(addem))
      if (nlibext.ge.1 ) then  
      write(UO,'(50A)')      "# FIR_LIB      : ",
     >(libext(i)(1:lnblnk(libext(i))),' ',i=1,nlibext)
      write(UO,'(A,1x,f8.2)')"# FIR_LMIN     : ",fir_lmin
      write(UO,'(A,1x,i10)') "# FIR_CONT     : ",IDINT(fir_cont)
      write(UO,'(A,1x,i10)') "# FIR_SCALE    : ",IDINT(fir_scale)
      write(UO,'(2A)')       "# FIR_FREESCALE: ",
     >                              fir_frsc(1:lnblnk(fir_frsc))
      write(UO,'(2A)')       "# FIR_SUBSTELLAR: ",
     >                              substar(1:lnblnk(substar))
      endif
      if (nlibphys .eq. 1) then 
      write(UO,'(50A)')      "# PHYS_LIB      : ",
     >                         libphys(1:lnblnk(libphys))
      write(UO,'(A,1x,i10)') "# PHYS_CONT     : ",IDINT(phys_cont)
      write(UO,'(A,1x,i10)') "# PHYS_SCALE    : ",IDINT(phys_scale)
      write(UO,'(A,1x,i10)') "# PHYS_NMAX     : ",phys_max
      endif
      write(UO,'(A,500(f6.3,1x))')
     >                    "# ERR_SCALE    : ",(min_err(i),i=1,nerr)
      write(UO,'(A,f6.3)') "# ERR_FACTOR   : ",fac_err
      write(UO,'(A,I10)')  "# BD_SCALE     : ",IDINT(bdscal)
      write(UO,'(A,I10)')  "# GLB_CONTEXT  : ",IDINT(gbcont)
c     AJOUTE pour chi2 check par bande  60,10
      write(UO,'(A,2f9.2)')"# CHI2_RM_BD   : ",levelChi2,levelMag
      write(UO,'(A,2(f8.3,1x))')"# Z_RANGE      : ",zmin_gal,zmax_gal
      write(UO,'(A,2(f8.3,1x))')"# EBV_RANGE  : ",ebvmin,ebvmax
      write(UO,'(A,f6.3)') "# DZ_WIN       : ",dz_win
      write(UO,'(A,f6.3)') "# MIN_THRES    : ",min_thres
      write(UO,'(A,2f8.3)')"# MASS_SCALE   : ",lmasi,lmass 
      write(UO,'(A,2f8.3)')"# MAG_ABS      : ",magabsl(1),magabsl(2)
      write(UO,'(A,I10)')  "# MAG_REF      : ",babs
      write(UO,'(A,500(f6.2,1x))')
     >                    "# ZFORM (model): ",(zform(i),i=1,nmod)
      write(UO,'(A,3(I10,1x))')  "# NZ_PRIOR     : ",bp,bp_B,bp_I 
      write(UO,'(2A)')     "# Z_INTERP     : ",zintp(1:lnblnk(zintp))
      write(UO,'(A,500(f6.3,1x))')
     >                     "# PROB_INTZ    : ",(int_pdz(i),i=1,npdz)
c
      if (nzmax.ge.1) then
       write(UO,'(A,500(I6,1x))')
     >                     "# ZMAX_FILT    : ",(zmfilt(i),i=1,nzmax)
       write(UO,'(A,500(f6.3,1x))')
     >                     "# MAG_MAGLIM   : ",(zmlim(i),i=1,nzmax)      
      endif
c
      if (nmeth.ge.1) then 
        write(UO,'(A,500(I4,1x))') 
     >   "# MABS_METHOD  : ",(method(i),i=1,nmeth)
        write(UO,'(A,500(I10,1x))') 
     >   "# MABS_CONTEXT : ",(magabscont(i),i=1,nmeth)    
        write(UO,'(A,500(I4,1x))') 
     >   "# MABS_REF     : ",(bapp(i),i=1,nmeth)    
      endif
      test=0
      do k=1,nmeth
        if (method(k).eq.4 .and.test.eq.0) then
          test=1 
          write(UO,'(A,500(I4,1x))')   "# MABS_FILT    : ",
     >  (bappOp(i),i=1,nbBinZ)
          write(UO,'(A,500(f6.3,1x))')"# MABS_ZBIN    : ",
     >  (zbmin(i),zbmax(i),i=1,nbBinZ)
        endif
      enddo

      if (nbshift.gt.0) write(UO,'(A,500(f7.3,1x))')
     > "# APPLY_SYSSHIFT: ",(shift(i),i=1,nbshift)
      write(UO,'(2A)')     "# AUTO_ADAPT   : ",
     >                      autoadapt(1:lnblnk(autoadapt))
      if (autoadapt(1:1) .eq. "Y") then
       write(UO,'(2A)')     "# ERROR_ADAPT  : ",
     >                      adapterror(1:lnblnk(adapterror))
       write(UO,'(A,3(I10,1x))')
     >                    "# ADAPT_BAND   : ",fl_auto,fl1,fl2
       write(UO,'(A,2(f6.2,1x))')"# ADAPT_LIM    : ",auto_thresmin,
     > auto_thresmax
       write(UO,'(A,I10)') "# ADAPT_POLY   : ",degre
       write(UO,'(A,I10)') "# ADAPT_METH   : ",meth_ada
       write(UO,'(A,I10)') "# ADAPT_CONTEXT: ",IDINT(adcont)
       write(UO,'(A,2(f6.3,1x))') "# ADAPT_ZBIN   : ",adzmin,adzmax
       write(UO,'(A,2(I8,1x))')   "# ADAPT_MODBIN : ",admmin,admmax
      endif
c
      write(UO,'(2A)') "# ZFIX         : ",zfix(1:lnblnk(zfix))
      write(UO,'(2A)') "# SPEC_OUT     : ",outsp(1:lnblnk(outsp))
      write(UO,'(2A)') "# CHI2_OUT     : ",outchi(1:lnblnk(outchi))
c
      write(UO,'(2A)') "# PDZ_OUT      : ",outpdz(1:lnblnk(outpdz))
      if (outpdz(1:4).ne."NONE") then
        write(UO,'(2A)')          "# PDZ_FILE     : ",
     >                   pdz_file(1:lnblnk(pdz_file))
        write(UO,'(2A)') "# PDZ_ZPH     : ",pdz_zph(1:lnblnk(pdz_zph))
        write(UO,'(2A)') "# PDZ_MOD     : ",pdz_mod(1:lnblnk(pdz_mod))
        write(UO,'(A,500(A,1x))')    "# PDZ_MABS     : ",
     >                   (pdz_abs(i)(1:lnblnk(pdz_abs(i))),i=1,nfabs)
        write(UO,'(A,500(1x,I4))')"# PDZ_MABS_FILT: ",
     >                   (pdz_fabs(i),i=1,nfabs)
c        write(UO,'(A,100(1x,I4))')"# PDZ_MABS_CONT: ",
c     >                   (pdz_cont(i),i=1,nfabs)
c        write(UO,'(A,100(1x,I4))')"# PDZ_MABS_METH: ",
c     >                   (pdz_meth(i),i=1,nfabs)

      endif       
c
      write(UO,'(2A)') "# FAST_MODE    : ",fastmod(1:lnblnk(fastmod))
      if (fastmod(1:1).eq."Y" .or. fastmod(1:1).eq."y") then
       write(UO,'(A,I10)') "# COL_NUM      : ",numcol
       write(UO,'(A,f6.2)')"# COL_SIGMA    : ",sigcol
       write(UO,'(2A)')    "# COL_SEL      : ",sel(1:lnblnk(sel))
      endif   
      write(UO,'(A)')  "#######################################"


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c READING INPUT LIBRARIES reclmax
      gallib= ' '
      qsolib= ' '
      starlib=' '
      write(UO,*) ' ' 
      write(UO,'(A)') "  >>>   reading input librairies ..."
c      write(UO,'(A,a1,$)') "reading input librairies ...",char(13)
c      call flush(UO)
      call read_lib(colib,numlib,inlib,
     >zstep,zmax,dz,colibmax,typlib,
     >modlib,extlib,ebvlib,zlib,agelib,reclib,
     >maglib,klib,emlib,emlines,ldustlib,
     >nzf,zf,zflib,
     >starlib,qsolib,gallib,physpara,
     >extlaw,nextlaw,ebv,nebv,modext,
     >h0,om0,l0,magtyp,valf,imagm,filters,nmodg,
     >fobs,minkcol)      
c    compute age of the universe at a given z
      write(UO,'(A)') "  ... computing Tuniv & Dist_Mod vs z "
      do i= 1,colibmax
         dmlib(i)  = -999. 
         ageuniv(i)=  20.e9
         z=zlib(i)
         if (typlib(i).le.2) then 
           dmlib(i) = funz(z,h0,om0,l0)
         endif
         if (typlib(i).eq.1 .and. agelib(i).ge.1.e4 
     >                      .and. agelib(i).lt.20.e9) then 
            ageuniv(i) = timy(z,h0,om0,l0)
c           write(*,*) typlib(i),zlib(i),agelib(i),ageuniv(i)
         endif
      enddo
c
c     typlib : 1 for GAL  / 2 for QSO / 3 for STAR
c
      if (lnblnk(catmag).ne.0 .and. catmag(1:5).ne.magtyp(1:5)) then
         write(UO,*) '  Magnitude types between libraries '
         WRITE(UO,*) '  and INPUT catalog are different   '
         stop
      endif    
c
      write(UO,*) '     number of record in libraries: ',colibmax
      write(UO,'(3A,1x,I9,A,I9)') '    number of models in ',
     > gallib(1:lnblnk(gallib)),' : ',nmodg,' & filters : ',imagm
c 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c READING FIR library
      if (nlibext.ge.1) then  
         write(UO,*) ' ' 
         write(UO,'(A)') "  >>>   reading FarIR librairies ..."
         call read_libfir(libext,nlibext,libfir,inlib,
     >    nfir,liblfir,modlfir,zlfir,lumlfir,reclfir,maglfir,klfir,
     >    zstep,zmax,dz,h0,om0,l0,magtyp,imagm)
      endif
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  COMPUTING THE FILTER CHARACTERISTICS
c      write(UO,'(A)') "#######################################"
      write(UO,*) ' ' 
      write(UO,'(A)') "  >>>   checking  filter characteristics..."    
c      write(UO,'(A,a1,$)') "   filter characteristics ...",char(13)
c      call flush(UO)
      call zeropoint(filters,zp,abcor,flmoy,flwidth,fcorr,imag)
      if (imag.ne.imagm) then
         write(UO,*) 'WARNING : Number of filters is not equal between'
         write(UO,*) filters(1:lnblnk(filters)),' and '
     >             ,colib(1)(1:(lnblnk(colib(1))))
         write(UO,*) ' check in ',config(1:lnblnk(config)),
     >                 ' the file name in FILTER_FILE '
         stop
      endif
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c            FINAL  CHECKS  ON   LIBRARIES
ccc  Check if appropriated filters for the use of FIR Library
c         Need at least (!!) filters at Lbda_mean > fir_lmin (um)  
      if (nlibext.ge. 1) then 
          k = 0 
          do i = 1,imag
             if ((flmoy(i)/10000.) .gt. fir_lmin)  k = k +1
          enddo
          if (k.eq.0) then
            nlibext=0      
            libext(1)='NONE'
            write(UO,*) '                                           '
            write(UO,*) '       !!!  WARNING !!!                    '
            write(UO,*) '  NO FIR LIBRARY USED because              '
            write(UO,*) '  all filters have Lbda<',fir_lmin
            write(UO,*) '   -->  FIR LIBRARY turns off   !!         '
            write(UO,*) '                                           '
          endif 
          nbufir=0
          nbsfir=0
          do k = 1, imagm
            busfir(k)=0
c    context for Band used for Chi2
            if (fir_cont.le.0) then      
               if ((flmoy(k)/10000.).ge.fir_lmin)  busfir(k)=1
            else
               busfir(k)=bdincl(k-1,fir_cont,imagm-1)
            endif
            if (busfir(k).eq.1) nbufir=nbufir+1
c    context for Band used for scaling 
            if (fir_scale.le.0) then      
               if ((flmoy(k)/10000.).ge. fir_lmin .and. busfir(k).eq.1) 
     >               bscfir(k)=1
            else
               bscfir(k)=bdincl(k-1,fir_scale,imagm-1)
               if (busfir(k).eq.0) bscfir(k)=0
            endif
            if (bscfir(k).eq.1) nbsfir=nbsfir+1
          enddo 
c
          if (nbsfir.eq.0 .OR.  nbufir.eq.0 ) then 
            nlibext=0      
            libext(1)='NONE'
            write(UO,*) '                                  '
            write(UO,*) '       !!!  WARNING !!!           '
            write(UO,*) '  FIR BAND USED FOR SCALING       '
            write(UO,*) '  has a  Lbda<',fir_lmin
            write(UO,*) '  -->  FIR LIBRARY turns off   !! '
            write(UO,*) '                                  '
          else 
            write(UO,*) ' FIR analysis based on the '
            write(UO,*) '  following context per band :      ' 
            write(UO,*) '   ',nbsfir,' bands for scaling with :'
            write(UO,*) '   ',(bscfir(k),k=1,imagm)
            write(UO,*) '   ',nbufir,' bands for Chi^2 with :'
            write(UO,*) '   ',(busfir(k),k=1,imagm)
          endif
c        
      endif
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccc    Check The PHYS library 
      if  (nlibphys.eq.1) then 
          call check_libphys(libphys,nlibphys,libphys_sed,libppara,
     >      zrecp,zrecpi,zrecps,nzrecp,
     >      h0,om0,l0,magtyp,imagm) 
c
         write(UO,'("Redshift records: ",I6,1x,2(f6.3,1x,2(I12,1x)))') 
     >    nzrecp,
     >    zrecp(1),zrecpi(1),zrecps(1),
     >    zrecp(nzrecp),zrecpi(nzrecp),zrecps(nzrecp)
c
        do k = 1,50
           nppara(k)=0
        enddo
      endif
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccc  check if Emission lines included in library in case addem=YES
      if (addem(1:1) .eq. 'Y') then 
         if (emlines(1:1) .ne. 'Y') then 
            addem='NO'
            write(UO,*) '                                           '
            write(UO,*) '       !!!  WARNING !!!                    '
            write(UO,*) '  Emission Lines cannot be applied because '
            write(UO,*) '  library does not include emission lines  '
            write(UO,*) '   -->  ADD_EMLINES turns off   !!         '
            write(UO,*) '                                           '
         endif
      endif
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccc    PREPARE THE OUPUT FORMAT           
      call prep_out(cattyp,cat,config,outpara,imagm,
     >  wpara,iwpara,npara,nppara)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c scaling errors check
      if (nerr.ne.imag) then
         do i = 1,imag
           min_err(i) = -1.
         enddo
      endif
c   
ccccccccccccccccccccccccccccccccccccccccccccccccc
c  Check zform_min keyword size with number of models
      if (nmod.ne.nmodg) then
         do i = 1, nmodg
            zform(i) = 0
         enddo
      endif
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   READ EXTINCTION AND OPACITY  if Output spectra = Y 
      if (outsp(1:1).eq.'Y' .or. addem(1:1) .eq.'Y') then
        if (addem(1:1) .eq. 'Y') then 
c   read UV filter for Emission lines scaling
          file = zpdir(1:lnblnk(zpdir)) //'/filt/galex/NUV.pb'
          open(1,file=file,status='old',err=56)
          read(1,*) 
          j=0
          do while (.true.)
            j=j+1
            read(1,*,end=14)  lambf_UV(j),repf_UV(j)
          enddo
 14       imuv=j-1  
          close(1)
        endif 
c 
        do j = 1, nextlaw
           if (extlaw(j)(1:4).ne.'NONE') then 
c  read extinction law 
             file = zpdir(1:lnblnk(zpdir)) //'/ext/' 
     >              // extlaw(j)(1:lnblnk(extlaw(j)))
             open(1,file=file,status='unknown',err=56)
             i=0
             do while (.true.)
               i=i+1
               read(1,*,end=11) extic(j,1,i),extic(j,2,i)
             enddo
 11          iext(j)=i-1
             close(1)
           else
             iext(j)=1
             extic(j,1,1)=0.
             extic(j,2,1)=0.
           endif 
           if (addem(1:1) .eq. "Y") then 
              iextinc=iext(j)
              do i = 1,iextinc
                 extinc(1,i)=extic(j,1,i)
                 extinc(2,i)=extic(j,2,i)
              enddo
              call ext_emlines(extinc,iextinc,
     >              lambf_UV,repf_UV,imuv,ext_em)
              do i = 1, 7
                 aext_lb(j,i)=ext_em(i)
c                 write(UO,*) j,i,aext_lb(j,i)
              enddo
           endif
        enddo 
        if (outsp(1:1).eq.'Y') then
c  reading file with  extragalacitic opacity
           file = zpdir(1:lnblnk(zpdir)) // '/opa/OPACITY.dat'
           open(1,file=file,status='unknown',err=56)
           do i = 1,81
              read(1,'(A)') str 
              call val_string(str,paravc,test)
              fileop(i)=paravc(2)
           enddo
           close(1)
           do i = 1, 81
              file = zpdir(1:lnblnk(zpdir)) // '/opa/' 
     > // fileop(i)
              open(1,file=file,status='unknown',err=56)
              k = 0
              do while (.true.)
                 k = k + 1
                 read(1,*,end=12) opal(i,k), opat(i,k)
              enddo  
 12           iopa(i) = k - 1
              close(1)
           enddo
        endif             
      endif
ccccccccccccccccccccccccccccccccccccccccccccccccc
c     Apply systematic shift to library 
      if(nbshift.eq.imagm)then
       do i = 1, colibmax
         do k = 1, imagm
            maglib(k,i)=maglib(k,i)+shift(k)
         enddo
       enddo
      endif
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   CONVERSION from theoretical magnitudes to fluxes
      write(UO,'(A,a1,$)') "converting mag to flux ...",char(13)
      call flush(UO)
      do i = 1, colibmax
         do k = 1, imagm
            if (maglib(k,i) .gt. 99) then 
                maglibf(k,i) = 0.
            else
             if (magtyp(1:1).eq.'A') 
     >              maglibf(k,i)=10**(-0.4*(maglib(k,i)+48.59))
             if (magtyp(1:1).eq.'V')
     >              maglibf(k,i)=10**(-0.4*(maglib(k,i)-zp(k)))
            endif
c       for AUTO-ADAPT
            maglibfini(k,i)=maglibf(k,i)
            maglibini(k,i) =maglib(k,i)
         enddo          
      enddo   
c     for FIR librries
      if (nlibext.ge.1) then
        do i = 1, nfir
           do k = 1, imagm
              if (maglfir(k,i) .gt. 90) then 
                  maglfir(k,i) = 0.
              else
                 if (magtyp(1:1).eq.'A') 
     >              maglfir(k,i)=10**(-0.4*(maglfir(k,i)+48.59))
                 if (magtyp(1:1).eq.'V')
     >              maglfir(k,i)=10**(-0.4*(maglfir(k,i)-zp(k)))
              endif               
           enddo          
        enddo   
      endif
c
cccccccccccccccccccccccccccccccccccccccccccccccc
c   Sort according to colors 
      if (fastmod(1:1).eq."Y") then
        write(UO,'(A)') "sorting the colors..."
        call flush(UO)
        call sort_col(maglibf,imagm,colibmax,numcol,fsort,f_index)
      endif  
ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   Add emission line flux according to the UVabs
      if (addem(1:1) .eq. 'Y') then 
         frac(1)=0.5
         frac(2)=1.
         frac(3)=2.
      endif
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c opening OUTPUT FILES  
      open(4,file=outf(1:lnblnk(outf)),status='unknown')
      if (outpdz(1:4).ne."NONE") then
        open(43,file=pdz_zph(1:lnblnk(pdz_zph)),status='unknown') 
        open(44,file=pdz_file(1:lnblnk(pdz_file)),status='unknown')  
        open(45,file=pdz_mod(1:lnblnk(pdz_mod)),status='unknown')  
        do i = 1,nfabs
          ufabs=45+i 
          open(ufabs,file=pdz_abs(i)(1:lnblnk(pdz_abs(i))),
     >          status='unknown')  
        enddo
      endif
c  Header for Output file 
      write(4,'(A)')   "#######################################"
      write(4,'(A)')   "# PHOTOMETRIC REDSHIFT with OPTIONS   #"
      write(4,'(A)')   "#          Input catalog              #" 
      write(4,'(2A)')  "# CAT_IN      : ",cat(1:lnblnk(cat))
      write(4,'(A,2(I10,1x))') "# CAT_LINES     : ",rowmin,rowmax
      write(4,'(2A)')  "# INP_TYPE    : ",typm(1:lnblnk(typm))
      write(4,'(2A)')  "# CAT_MAG     : ",catmag(1:lnblnk(catmag))
      write(4,'(A,2x,I4)') "# CAT_FMT[0:MEME 1:MMEE]: ",cat_fmt
      write(4,'(A)')   "#          Input library              #"    
      write(4,'(2A)')  "# FILTER_FILE : ",filters(1:lnblnk(filters))
      write(4,'(100A)')"# ",magtyp(1:5),(valf(j)(1:lnblnk(valf(j)))
     >             ,' ',j=1,imagm)
      write(4,'(A,1x,100(f8.3,1x))') "# ",(abcor(j),j=1,imagm)
      write(4,'(8(A))')"# ZPHOTLIB    : ",
     > (colib(i)(1:lnblnk(colib(i))),' ',i=1,numlib)
      write(4,'(8(A))')"# using       : ",gallib(1:lnblnk(gallib))
     >,' ',qsolib(1:lnblnk(qsolib)),' ',starlib(1:lnblnk(starlib))
      write(4,'(2A)')  "# ADD_EMLINES  : ",addem(1:lnblnk(addem))
      write(4,'(A,1x,3f6.2)') "# Z_STEP      : ",zstep,zmax,dz
      write(4,'(A,2f9.2)')    "# CHI2_RM_BD  : ",levelChi2,levelMag
      write(4,'(A,2(f8.3,1x))')"# Z_RANGE     : ",zmin_gal,zmax_gal
      write(4,'(A,2(f8.3,1x))')"# EBV_RANGE   : ",ebvmin,ebvmax
      write(4,'(2A)')         "# Z_INTERP    : ",zintp(1:lnblnk(zintp))
      write(4,'(A,1x,3f6.2)') "# COSMOLOGY   : ",h0,om0,l0
      if (nextlaw .ge. 1) then 
        write(4,'(11(A,2x))')         "# EXTINC_LAW  : ",
     >                 (extlaw(i)(1:lnblnk(extlaw(i))),i=1,nextlaw)
        write(4,'(A,1x,500f6.3)')"# EB_V        : ",(ebv(i),i=1,nebv)
        write(4,'(A,1x,20(I6,1x))')
     >                  "# MOD_EXTINC  : ",(modext(i),i=1,2*nextlaw)
      endif 
      if (nlibext .ge. 1) then 
        write(4,'(A,50(A,2x))')   "# FIR LIBRARY : ",
     >                 (libfir(i)(1:lnblnk(libfir(i))),i=1,nlibext)
        write(4,'(2A)')       "# FIR_FREESCALE: ",
     >                              fir_frsc(1:lnblnk(fir_frsc))
        write(4,'(2A)')       "# FIR_SUBSTELLAR: ",
     >                              substar(1:lnblnk(substar))
        write(4,'(A,1x,f8.2)')"# FIR_LMIN      : ",fir_lmin

        write(4,'(A,1x,2(i10,2x))') 
     >                    "# FIR_CONT     : ",IDINT(fir_cont),nbufir
        write(4,'(A,1x,2(i10,2x))')
     >                    "# FIR_SCALE    : ",IDINT(fir_scale),nbsfir
      endif
      if (nlibphys.eq.1) then 
         write(4,'(50A)')      "# PHYS_LIB      : ",
     >                         libphys(1:lnblnk(libphys))
         write(4,'(A,1x,i10)') "# PHYS_CONT     : ",IDINT(phys_cont)
         write(4,'(A,1x,i10)') "# PHYS_SCALE    : ",IDINT(phys_scale)
         write(4,'(A,1x,i10)') "# PHYS_NMAX     : ",phys_max
      endif 
      write(4,'(2A)') "# FAST_MODE   : ",fastmod(1:lnblnk(fastmod))
      if (fastmod(1:1).eq."Y") then
         write(4,'(A,I10)')      "# COL_NUM     : ",numcol
         write(4,'(A,f6.2)')     "# COL_SIGMA   : ",sigcol
         write(4,'(2A)')         "# COL_SEL     : ",sel(1:lnblnk(sel))
      endif
c
      write(4,'(A,500(f6.3,1x))') "# ERR_SCALE   : ",(min_err(i),
     >  i=1,nerr)
      write(4,'(A,f6.3)') "# ERR_FACTOR  : ",fac_err
      write(4,'(A)') "#          Options                    #"
      write(4,'(A,I8)')   "# BD_SCALE    : ",IDINT(bdscal)
      write(4,'(A,I12)')  "# GLB_CONTEXT : ",IDINT(gbcont)
      write(4,'(A,f6.3)') "# DZ_WIN      : ",dz_win
      write(4,'(A,f6.3)') "# MIN_THRES   : ",min_thres
      write(4,'(A,2(f6.2,1x))') "# MASS_SCALE  : ",lmasi,lmass 
      write(4,'(A,2(f8.2,1x))') "# MAG_ABS     : ",(magabsl(i),i=1,2)
      write(4,'(A,1x,I8)')      "# MAG_REF     : ",babs
      write(4,'(A,500(f6.3,1x))')
     >                    "# PROB_INTZ   : ",(int_pdz(i),i=1,npdz)
c
      if (nmod .eq. nmodg) then
         write(4,'(A,100(f6.2,1x))') "# ZFORM_MIN(model): ",
     > (zform(i),i=1,nmod)
      endif   
c
      if (nzmax.ge.1) then
       write(4,'(A,500(I6,1x))')
     >                     "# ZMAX_FILT    : ",(zmfilt(i),i=1,nzmax)
       write(4,'(A,500(f6.3,1x))')
     >                     "# MAG_MAGLIM   : ",(zmlim(i),i=1,nzmax)      
      endif
c
      if (nmeth.ge.1) then 
        write(4,'(A,500(I4,1x))') 
     >                    "# MABS_METHOD : ",(method(i),i=1,nmeth)
        write(4,'(A,500(I10,1x))') 
     >                    "# MABS_CONTEXT: ",(magabscont(i),i=1,nmeth)
        write(4,'(A,500(I4,1x))') 
     >                    "# MABS_REF    : ",(bapp(i),i=1,nmeth)    
      endif
      test=0
      do k=1,nmeth
        if (method(k).eq.4 .and.test.eq.0) then
          test=1 
          write(4,'(A,500(I4,1x))')   "# MABS_FILT    : ",
     >  (bappOp(i),i=1,nbBinZ)
          write(4,'(A,500(f6.3,1x))') "# MABS_ZBIN    : ",
     >  (zbmin(i),zbmax(i),i=1,nbBinZ)
        endif
      enddo
ccccc
      if (nbshift.gt.0) write(4,'(A,100(f7.3,1x))')
     >                     "# APPLY_SYSSHIFT: ",(shift(i),i=1,nbshift)
      write(4,'(2A)')      "# AUTO_ADAPT  : ",
     >                      autoadapt(1:lnblnk(autoadapt))
      if (autoadapt(1:1) .eq. "Y") then
        write(4,'(2A)')    "# ERROR_ADAPT  : ",
     >                      adapterror(1:lnblnk(adapterror))
        write(4,'(A,3(I10,1x))')
     >                     "# ADAPT_BAND   : ",fl_auto,fl1,fl2
        write(4,'(A,2(f6.2,1x))')
     >                     "# ADAPT_LIM    : ",auto_thresmin,
     >  auto_thresmax
        write(4,'(A,I10)') "# ADAPT_POLY   : ",degre
        write(4,'(A,I10)') "# ADAPT_METH   : ",meth_ada
        write(4,'(A,I10)') "# ADAPT_CONTEXT: ",IDINT(adcont)
        write(4,'(A,2(f6.3,1x))') "# ADAPT_ZBIN  : ",adzmin,adzmax
        write(4,'(A,2(I8,1x))')   "# ADAPT_MODBIN: ",admmin,admmax
      endif
c
      write(4,'(A,1x,3(I8,1x))') "# NZ_PRIOR    : ",bp,bp_B,bp_I 
      write(4,'(2A)')      "# ZFIX        : ",zfix(1:lnblnk(zfix))
      write(4,'(2A)')      "# SPEC_OUT    : ",outsp(1:lnblnk(outsp))
      write(4,'(2A)')      "# CHI2_OUT    : ",outchi(1:lnblnk(outchi))
c
      write(4,'(2A)')      "# PDZ_OUT     : ",outpdz(1:lnblnk(outpdz))
      if (outpdz(1:4).ne."NONE") then
        write(4,'(2A)')          "# PDZ_file    : ",
     >                   pdz_file(1:lnblnk(pdz_file))
        write(4,'(2A)') "# PDZ_ZPH     : ",pdz_zph(1:lnblnk(pdz_zph))
        write(4,'(2A)') "# PDZ_MOD     : ",pdz_mod(1:lnblnk(pdz_mod))
        write(4,'(A,100(A,1x))')    "# PDZ_MABS    : ",
     >                   (pdz_abs(i)(1:lnblnk(pdz_abs(i))),i=1,nfabs)
        write(4,'(A,100(1x,I4))')"# PDZ_MABS_FILT: ",
     >                   (pdz_fabs(i),i=1,nfabs)
      endif       
c
      write(4,'(2A)')      "# CAT_OUT     : ",outf(1:lnblnk(outf))
      write(4,'(2A)')      "# PARA_OUT    : ",outpara(1:lnblnk(outpara))
      cr_date=fdate()
      write(4,'(2A)') "# CREATION_DATE: ",cr_date(1:lnblnk(cr_date)) 
      write(4,'(A)')  "# Output format                       #"
      val1 = DINT(DBLE(iwpara)/5.)
      if (MOD(iwpara,5).eq.0) then
          k = IDINT(val1)
      else
          k = IDINT(val1) + 1
      endif
      do i = 1,k
         imin = (i-1)*5 + 1
         imax = imin + 4
         if (imax.gt.iwpara) imax = iwpara
        write(4,'(A,1x,5(A,1x,i3," , "))') "# ",
     >  (wpara(j)(1:lnblnk(wpara(j))),npara(j),j=imin,imax)

      enddo
      write(4,'(A)') "#######################################"
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  convert mass in dexp if used
      if (lmasi.gt.0 .and. lmass.gt.0 ) then
         lmasi=10**(lmasi)
         lmass=10**(lmass)
      endif   
c Initialisation of chi2 evolution with z and chimin
      if (zmax.le.6.0)  then 
        chimax   = idnint(zmax/zstep) + 1
        chitrans = chimax
      else
        chitrans = idnint(6./zstep) + 1 
        chimax   = chitrans  + idnint((zmax-6.)/dz)
      endif
c Compute the distance modulus for mag_abs
      do k = 1,chimax
        if (k.le.chitrans) then
           chi(1,k) = zstep*(k-1)
        else 
           chi(1,k) = zstep*(chitrans-1) + dz*(k-chitrans)
        endif
        z=chi(1,k)
        dist_mod(k)=funz(z,h0,om0,l0)
      enddo 
c used DM at z=dz/20 for z=0 instead of DM=0 bacause of rescaling dm 
c   only use for computation of abs_mag (the Mag_abs prior)
      funz0=funz(zstep/20.d0,h0,om0,l0)
c      write(*,'(A,2x,f6.3,2x,e12.6)') 'DM(dz/20)',zstep/10.,funz0 
c save the z bin for PDZ_OUT 
      if (outpdz(1:4).ne."NONE") then
        write(43,'(500(f8.3,1x))') (chi(1,k),k=1,chimax)
      endif
c Number of measured objects
      do k = 1,3 
        nmeas(k)=0
      enddo  
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     For the case of method 4 in absolute magnitude
c     The imposed observed filter depends on redshift
      test=0
      do j = 1,nmeth
        if(method(j).eq.4 .and. test.eq.0) then
          test=1
          do i = 1, chimax
           do k=1,nbBinZ  !loop en redshift bin
            if(chi(1,i).ge.zbmin(k).and.chi(1,i).lt.zbmax(k)) then
c         If z include in redshift bin k, use the correspondant k filter
             do l=1,imagm
              do m=1,imagm
                fobs4(i,l,m)=bappOp(k)
              enddo
             enddo 
            endif
           enddo
          enddo
        endif
      enddo
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Initialisation of AUTO-ADAPT
      do k=1,imagm
       a0(k)=0.d0
       a1(k)=0.d0
       a2(k)=0.d0
       a3(k)=0.d0
      enddo
      x=0.d0
      x2=0.d0
      x3=0.d0
      model=0
      chiin=10.d10 
      iteration=0 
      realise=0
      res_best=20000.d0
      if (autoadapt(1:1).eq.'Y') then
      write(UO,'(A)') "##############################################"
        write(UO,'(A,1x,I4)') " --> Starting AUTO-ADAPT with method:",
     >   meth_ada
        open(41,file='minuit.dat',status='unknown')
        open(42,file=outf(1:lnblnk(outf))//'.corr',status='unknown')
      endif
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     REDO from this point if AUTO-ADAPT didn't converge
 6    continue
      if (autoadapt(1:1).eq.'Y') then
c        Reinitialize the number of objects
         do k = 1,3 
           nmeas(k)=0
         enddo 
         iteration=iteration+1
         if (realise.lt.1) 
     >      write(UO,'(A,I4)') " --> Iteration :",iteration
c        Stop the iteration if more than 10 and take the values at the best point
         if(iteration.ge.10.and.realise.lt.2) then
            write(UO,*)"MORE THAN 10 ITERATIONS !!"
            write(UO,*)"STOP TRAINING !!"
            realise=2
c           take the best value of iteration
            write(UO,*)"Take the value of iteration :",iter_best
             do k=1,imagm
               a0(k)=a0best(k)
               a1(k)=a1best(k)
               a2(k)=a2best(k)
               a3(k)=a3best(k)
               min_err(k)=min_errbest(k)
            enddo
         endif
c        loop on library and apply the correction determined with auto-adapt
         do i = 1, colibmax
c         Determine the correction to apply according to the method 
            x=0.d0  
            if (meth_ada.eq.1) then 
c         Predicted color
               if(dabs(maglibini(fl1,i)).le.80.and.
     .            dabs(maglibini(fl2,i)).le.80) then
                  x=maglibini(fl1,i)-maglibini(fl2,i)
               endif
            elseif (meth_ada.eq.2) then 
c           Redshift
               x=zlib(i) 
            elseif (meth_ada.eq.3) then          
c            Model
               x=dble(modlib(i))  
            endif
c         Apply correction to the library
            if(degre.ge.3)x2=x**2.d0
            if(degre.ge.4)x3=x**3.d0
            do k=1,imagm 
               corr=a0(k)+a1(k)*x+a2(k)*x2+a3(k)*x3
               maglib(k,i) =maglibini(k,i)+corr
               maglibf(k,i)=maglibfini(k,i)*10.d0**(-0.4*corr)
            enddo
         enddo   
      endif
ccccccccccAUTO-ADAPT
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  OPEN THE PHOTOMETRIC CATALOG  + OUTPUT FILE 
      irecz0 =1
      pweight=1
      cont   =0.d0 
      zs     =-99.d0
      str_inp= ' '
      nobj   = 0
      ngals_ada=1
      open(90,file=cat,status='unknown')      
      do while (.true.)                      ! --> closed at the END
         read(90,'(A)',end=8) str 
         call val_string(str,paravc,test)
         if (paravc(1)(1:1) .eq. '#') goto 1 
         read(paravc(1),'(i10)') spec
         do k = 1,imagm
            if (cat_fmt .eq. 0) j = 2*k
            if (cat_fmt .eq. 1) j = k+1
            call check_float(paravc(j),str_ch)
            read(str_ch,'(E24.12)') ab(k)
            if (cat_fmt .eq. 0) j=j+1
            if (cat_fmt .eq. 1) j=j+imagm
            call check_float(paravc(j),str_ch)
            read(str_ch,'(E24.12)') sab(k)
            aborig(k)=ab(k)
            saborig(k)=sab(k)
cccccccccccccccccccccccccccccccccccccccccccc
c  put original values in mag and errmag
            if (typm.eq.'M') then
               abo(k) = ab(k)
               sabo(k)= sab(k)
            else
              if (sab(k).lt.0)  sabo(k) = sab(k) 
              if (sab(k).ge.0)  then
                if (ab(k).gt.0) then 
                   sabo(k)=1.086*sab(k)/ab(k)
                else
                   sabo(k) = 2.0
                endif  
              endif
              if (ab(k).gt.0) then 
                 if (magtyp(1:1).eq.'A') abo(k)=-2.5*dlog10(ab(k))-48.59
                 if (magtyp(1:1).eq.'V') abo(k)=-2.5*dlog10(ab(k))+zp(k)
              else
                 abo(k)=999.0
              endif
            endif
         enddo    
ccccccccccccccccccccccccccccccccccccccccc
         if (cattyp(1:4).eq.'LONG') then
            j=2*imagm+2
            if (gbcont.eq.-1) then 
c               read(paravc(j),'(E24.0)') conti
               read(paravc(j),'(I24)') conti
               cont = DBLE(conti)
            else
               cont=gbcont
            endif   
            j=j+1
            call check_float(paravc(j),str_ch)
            read(str_ch,'(E20.9)') zs
            j=j+1
c read Lir prior luminosity in Lo(Lo)
            if (lirprior(1:1).eq."Y") then 
              call check_float(paravc(j),str_ch)
              read(str_ch,'(E20.9)') lirinp
              j=j+1
              call check_float(paravc(j),str_ch)
              read(str_ch,'(E20.9)') dlirinp
              j=j-1
            endif
c read the rest of the string 
            str_inp = ' '
            do k = j,test
              str_inp = str_inp(1:lnblnk(str_inp)) // " " //
     >                 paravc(k)(1:lnblnk(paravc(k)))
            enddo
         else
           if (gbcont.eq.-1) then 
              cont = 0.d0
           else
              cont = gbcont
           endif   
         endif   
c         
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  read obs mag catalogue 
      nobj = nobj + 1
c  do only the objects in range : rowmin<row<rowmax
      if (rowmin.gt.0 .and.rowmax.ge.rowmin) then 
        if (nobj.lt. rowmin .OR. nobj.gt.rowmax ) goto 1
      endif
      pass = 0
      abs_mag=-99
      redoing=0
      redofit=0
c
c     Redo the fit if one band is out in chi2 or mag       
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  CONTEXT translated in used bands bused(k)=0 (NOT USED) or 1 (USED)
      nbused=0
      new_cont=0.d0
      nbus=0                        ! band used in Chi^2     (right context+err>0)
      nbul=0                        ! band used as upper lim (right cont but err<0)
      do k = 1, imagm
         if (cont.eq.0) then         ! by default if it is not read
            bused(k)=1
         else
            bused(k)=bdincl(k-1,cont,imagm-1)
         endif
c  reject band if forbiden context defined 
         if(contforb.gt.0.and.
     >       bdincl(k-1,contforb,imagm-1).eq.1) bused(k)=0
c  reject band if error and flux lower than 0 
         if (sab(k).lt.0 .and. ab(k).lt. 0) bused(k)=0
         if (bused(k).eq.1) nbused=nbused+1
         if (bused(k).eq.1) new_cont=new_cont+(2.d0**(DBLE(k)-1.d0))
         if (bused(k).eq.1 .and. sab(k).gt.0) nbus=nbus+1
         if (bused(k).eq.1 .and. sab(k).le.0) nbul=nbul+1
c    context for Band used for scaling 
         if (bdscal.eq.0) then 
            buscal(k) = bused(k)
         else
            if (bused(k).eq.1 ) then 
              buscal(k)=bdincl(k-1,bdscal,imagm-1)
            endif
         endif
c        write(*,*) bused(k),buscal(k)
      enddo 
cccccccccccccccccccc redo Chi2 if 1 band  with high contribution to global Chi2
 35   if (redofit.eq.1) then 
         nbused=0
         new_cont=0.d0
         nbus=0 
         nbul=0 
          do k = 1,imagm
            sab(k) = saborig(k)
            ab(k)  = aborig(k)
            if (bused(k).eq.1) nbused=nbused+1
            if (bused(k).eq.1) new_cont=new_cont+(2.d0**(DBLE(k)-1.d0))
            if (bused(k).eq.1 .and. sab(k).gt.0) nbus=nbus+1
            if (bused(k).eq.1 .and. sab(k).le.0) nbul=nbul+1
         enddo
      endif
c  
c    context when measuring the Mag abs 
      do k = 1,imagm
        macont=DBLE(magabscont(k))
        do j = 1,imagm 
          mbused(k,j)=0
          if (macont.le.0)  mbused(k,j)=bused(j)
          if (macont.gt.0.and.bused(j).eq.1.and.
     >       bdincl(j-1,macont,imagm-1).eq.1) mbused(k,j)=1
        enddo
      enddo   
cccccccccccccccccccccccccccccccccccccccccccccccccccc
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  Conversion from observed magnitude  to flux if it is the case  
c     Vega:  mi = -2.5 logfi + ZPi , AB:  mi = -2.5 logfi  - 48.59
c     fi = 10**-0.4(mi+zpi)     df =  f * dm/1.086
c     Re-scaling of errors in quadratic
      if (typm.eq.'M') then 
         do k = 1,imagm
            if(ab(k).le.-100)  ab(k)=-99.
            if(sab(k).le.-100) sab(k)=-99.
            if (magtyp(1:1).eq.'A') ab(k)=10**(-0.4*(ab(k)+48.59))
            if (magtyp(1:1).eq.'V') ab(k)=10**(-0.4*(ab(k)-zp(k)))
            if (sab(k).gt.0.0) then 
              if (min_err(k).gt.0) then 
                 sab(k) = dsqrt(sab(k)**2+min_err(k)**2)
              endif   
              sab(k) = ab(k)*sab(k)/1.086*fac_err
            endif
         enddo   
      else
         do k = 1,imagm
           if ( min_err(k).gt.0) then
             if (ab(k).gt.0.and.sab(k).gt.0.and.sab(k).ne.ab(k)) then
               sab(k)=1.086*sab(k)/ab(k)
               sab(k)=dsqrt(sab(k)**2+min_err(k)**2)
               sab(k)=ab(k)*sab(k)/1.086
             endif   
           endif   
           if (sab(k).gt.0) sab(k)=sab(k)*fac_err
         enddo      
      endif
c
cccccccccc AUTO-ADAPT   SELECT OBJECTS WITH ALL BANDS MEASURED
      if (autoadapt(1:1).eq.'Y'.and. realise.lt.1) then
         if (ab(fl_auto).le.0.d0) goto 1
         if (-2.5*dlog10(ab(fl_auto))-48.59.ge.auto_thresmax) goto 1
         if (-2.5*dlog10(ab(fl_auto))-48.59.le.auto_thresmin) goto 1
         if (sab(fl_auto).le.0.d0.or.bused(fl_auto).eq.0)     goto 1
         if (zs.lt.adzmin .OR. zs.gt.adzmax .OR. zs.ge.zmax)  goto 1 
         if (degre.gt.1) then 
           if (sab(fl1).le.0.d0.or.bused(fl2).eq.0) goto 1
           if (sab(fl1).le.0.d0.or.bused(fl2).eq.0) goto 1
         endif
      endif
c
cccccccccccccc  Minimizes the search box in color space ccccccccccccc
c          Extract record to be used according to colors 
      if (fastmod(1:1) .eq. "Y") then
        call sear_rec(ab,sab,imagm,bused,sigcol,numcol,sel,
     >              fsort,f_index,colibmax,reclist,numrec)
      endif
cccccccccccccccccccccccccccccccccccccccccccccccccc
 36   if (redoing.eq.1)   liblength=colibmax
      do k = 1,500         ! Number of secondary peaks
        pdz(k) = 0
        nb(k)  = 1 
      enddo
      do k = 1, 500
         dzpdz(k) =0
      enddo
      zbay=-99.
      zbayi=-99.
      zbays=-99.
      z68i=-99.
      z68s=-99.
      do k = 1,imagm       ! Number of filters 
        goodfilter(k) = -1
        kap(k)    = -999
        mabs(k)   = -999
        kapq(k)   = -999
        mabsq(k)  = -999
        magm(k)   = -999
      enddo 
      do i = 1,3
         zpdzi(i) = 0.
         zpdzs(i) = 0.
         probz(i) = 0.
      enddo   
      zml68i=zpdzi(1)
      zml68s=zpdzs(1)
      zml90i=zpdzi(2)
      zml90s=zpdzs(2)
      zml99i=zpdzi(3)
      zml99s=zpdzs(3)      
c
      if (outsp(1:1).eq.'Y') then    ! Init. spectra
        do k = 1, wmax
          wsp(k)=0.
          fsp(k)=0.
          wq(k) =0.
          fq(k) =0.
          wst(k)=0.
          fst(k)=0.
          do j = 1,5 
            fgal(j,k)=0.
            wgal(j,k)=0.
          enddo
        enddo
      endif
      do k = 1,chimax    !   chi with z = 0 --> zmax 
         chi(2,k) = 1.e9
         maxlz(k) = 0
         imasb(k) = -999
         extilb(k)= 0
         reclb(k) = 0
         extb(k)  = -99.
         ldustb(k)= -99. 
         ageb(k)  = -99.
         zb(k)    = -99.
         dmb(k)   = -99.
         zfb(k)   = -99.
         mag_absb(k)=-99.
         chibay(k)=0. 
         do j = 1,imagm
           kcorb(j,k)=-9999.
         enddo
      enddo
      do k = 1, 50
         physpbest(k)=-99.
      enddo
c
      do nlib = 1 , 3     !  best fittting for the 3 libraries
        chimin(nlib)   = 1.e10
        zmin(nlib)     = -99
        dmmin(nlib)    = -99
        recmin(nlib)   = -99
        imasmin(nlib)  = -999
        agemin(nlib)   = -99.
        extmin(nlib)   = -99.
        zfmin(nlib)    = -99.
        mag_abs(nlib)  = -99.
        mod_distb(nlib)= -99.
        reclmin(nlib)  = 0
        extilmin(nlib) = 0
        ldustmin(nlib) = -99.
      enddo
      if (fastmod(1:1) .eq. "Y")  then 
         liblength=numrec
      else
         liblength=colibmax
      endif
      if (redoing.eq.1)   liblength=colibmax
      mabsuv=99 
      nuvr=-99.
      
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c START BIG LOOP 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      do i = 1, liblength    ! library models (correspondance with goto 2 )
         if (fastmod(1:1) .eq. "Y" .AND. redoing.eq.0) then         
            nlib = typlib(reclist(i))           
            model= modlib(reclist(i))
            recl = reclib(reclist(i))
            extil= extlib(reclist(i))
            exti = ebvlib(reclist(i)) 
            z    = zlib(reclist(i))
            age  = agelib(reclist(i))
            tuniv= ageuniv(reclist(i))
            zfmod= zflib(reclist(i))
            ldust= ldustlib(reclist(i))
c            mod_dist=dmodlib(reclist(i))            
            do k = 1, imagm
              mag(k)  = maglibf(k,reclist(i))
              magm(k) = maglib(k,reclist(i))
              kcor(k) = klib(k,reclist(i))
              em(k)   = emlib(k,reclist(i))
            enddo          
         else
           nlib = typlib(i)
           model= modlib(i)
           recl = reclib(i)
           extil= extlib(i) 
           exti = ebvlib(i)
           ldust= ldustlib(i) 
           z    = zlib(i)
           age  = agelib(i)
           tuniv= ageuniv(i)
           zfmod= zflib(i)
c           mod_dist=dmodlib(i)
           do k = 1, imagm
             mag(k)  = maglibf(k,i)
             magm(k) = maglib(k,i)
             kcor(k) = klib(k,i)
             em(k)   = emlib(k,i)
           enddo   
         endif   
c         
c  keep the Chi2 for all para and z 
c        1  2   3    4    5     6    7   8   9      10   11  12   
c        z,age,ext-l,ebv,ldust, luv, lr, lk, ldust, mo, sfr  chi2
c         <-- mag_gal file --> <---- phys file        --->
         do k = 1,12 
            zchipara(k,i) = -9999
         enddo
         dm = 0  
c   z=0 for Mag abs 
         if (nlib.le.2 .and. z.le.1.e-5) irecz0=i   
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c    rejecting models 
c         if( zfix(1:1).eq.'Y' .AND. zs.ge.0 .AND. zs.lt.zmax .AND. 
c     >     nlib.eq.1 .AND. SNGL(DABS(z-zs)).gt.(SNGL(zstep/2.))) then 
         if( zfix(1:1).eq.'Y' .AND. zs.ge.0 .AND. zs.lt.zmax .AND. 
     >     nlib.eq.1 .AND. SNGL(DABS(z-zs)).gt.(SNGL(zstep))) then 
c  check if is running auto-adapt or not before moving to goto 2 
           if (autoadapt(1:1).eq.'N' .OR. realise.ge.2)  goto 2
         endif
         if(nlib.eq.1.and.(z.lt.zmin_gal.OR.z.gt.zmax_gal))      goto 2 
         if(ebvmin.ge.0 .AND.(exti.lt.ebvmin.OR.exti.gt.ebvmax)) goto 2
         if(autoadapt(1:1).eq.'Y'.and.realise.le.1.and.nlib.ne.1)goto 2
         if(autoadapt(1:1).eq.'Y'.and.realise.le.1 .and.  
     >       SNGL(DABS(z-zs)).gt.(SNGL(zstep/2.)) )              goto 2
         if(nlib.eq.1 .and. age.gt.tuniv)                        goto 2 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   Measurement of scaling factor dm only with (fobs>flim), dchi2/ddm = 0
         avmago = 0. 
         avmagt = 0. 
         nf = 0 
         do k = 1,imagm
           if (ab(k).gt.0.and.sab(k).gt.0.and.
     >         bused(k).eq.1) then
              if(buscal(k).eq.1)then
                 nf = nf + 1
                 avmago = avmago + ab(k)*mag(k)/sab(k)**2
                 avmagt = avmagt + mag(k)*mag(k)/sab(k)**2
             endif
           endif   
         enddo
         if (nf.le.1 .and. i.eq.1) then
             write(UO,*) spec,
     >            '  WARNING: No scaling 1 or zero band --> No z' 
             write(UO,'(I10,1x,200(f6.2,1x))') spec,(sab(k),k=1,imagm)
             goto 4          ! go to write output 
         else   
            if (avmagt.lt.1.d-60) goto 2
             dm = avmago/avmagt
         endif 
ccccccccccccccccccccccccccccc
c        MODEL  REJECTION  
         do k = 1,imagm 
           if (bused(k).eq.1.and.sab(k).le.0.and.
     >                    (dm*mag(k)).gt.ab(k)) goto 2
         enddo
c        Use Mass scaling for model rejection
         if (nlib.eq.1 .and. lmasi.gt.0 .and. lmass.gt.0 ) then
           if (dm.lt.lmasi.or.dm.gt.lmass)      goto 2
         endif
c        Use Age at z larger than Age for Zform
         zform2 = zform(model)
         if (nlib.eq.1 .and. zform2.ge.z .and. age.gt.1.e4) then 
c              tuniv  = timy(z,h0,om0,l0)      ! zinf -> z
              tzform = timy(zform2,h0,om0,l0) ! zinf -> zform
              tused  = tuniv-tzform           ! Age gal at z for a zform
              if (age.lt.tused) goto 2
         endif   
c        Abs Magnitude from model @ z=0 for rejection if babs defined 
         if (nlib.le.2) then  
           if (fastmod(1:1).eq."Y") then 
               if (babs.gt.0) abs_mag=maglib(babs,reclist(irecz0))
     >                                -2.5*dlog10(dm)
           else
               if (babs.gt.0) abs_mag=maglib(babs,irecz0)
     >                                -2.5*dlog10(dm)
           endif
           if (z.le.1.e-5) abs_mag=abs_mag-funz0  
c           write(*,*) z,irecz0,maglib(babs,irecz0),funz0,dm,
c     >            abs_mag,magabsl(1),magabsl(2) 
           if (nlib.le.1.and. babs.gt.0.and.
     >          magabsl(1).lt.0 .and. magabsl(2).lt.0) then 
               if (magabsl(1).le.magabsl(2)) then
                  if (abs_mag.lt.magabsl(1).OR.
     >                abs_mag.ge.magabsl(2))         goto 2
               elseif (magabsl(1).gt.magabsl(2)) then
                  if (abs_mag.lt.magabsl(2).OR.
     >                abs_mag.ge.magabsl(1))         goto 2
               endif
           endif
c         if (nlib.eq.1) write(*,*) '1: ',nlib,dm,nf,z,zs
           if (nlib .eq.1 .AND. addem(1:1) .eq. 'Y' ) then 
               nuvr=-2.5*(physpara(2,recl)-physpara(3,recl))   
               mabsuv= -2.5*physpara(2,recl) + 51.61
               mabsuv=mabsuv-2.5*dlog10(dm)
               mabsuv=mabsuv+aext_lb(extil,1)*exti
           endif
         endif  
ccccccccccccccccccccccccccccccccccccccccc
c  Chi2 weighted by Nz if prior used
         if (nlib.eq.1 .and. bp.gt.0 .and. bused(bp).eq.1 
     >        .and.sab(bp).gt.0) then
           if (magtyp(1:1).eq.'A') then 
              iab = -2.5*dlog10(ab(bp)) - 48.59
              color_rf=maglib(bp_B,irecz0)-maglib(bp_I,irecz0)
           elseif (magtyp(1:1).eq.'V') then 
               iab = -2.5*dlog10(ab(bp))+zp(bp) + abcor(bp) 
               color_rf=maglib(bp_B,irecz0)-maglib(bp_I,irecz0) 
     >                 +abcor(bp_B)-abcor(bp_I)
           endif
c             pweight=nzpriorVVDS5(iab,model,z)
             pweight=nzprior2(iab,color_rf,z)
         endif
cccccccccccccccccccccccccccccccccccccccccc
c  Add Emission line flux to mag based on UVabs if restframe (NUV-r)<4
         if(nlib.eq.1.and.addem(1:1).eq.'Y') then 
          if (nuvr.le.4 ) then 
            do k = 1,imagm
                em(k)=em(k)*10**(-0.4*(mabsuv+20))
            enddo
c               Loop on the emission lines
            fracMin=1.d0
            dmmi=dm
            chiMinEm=10.e9
c            Loop on the OIII and Lyalpha flux
            do j=1,3
                do k = 1,imagm
                   em2(k)=em(k)
                   if(((dabs(flmoy(k)-(5007.*(1.+z))).le.flwidth(k)).or.
     .                 (dabs(flmoy(k)-(4959.*(1.+z))).le.flwidth(k)).or.
     .                 (dabs(flmoy(k)-(1216.*(1.+z))).le.flwidth(k)).or.
     .              (dabs(flmoy(k)-(4861.*(1.+z))).le.flwidth(k))))then
                        em2(k)=frac(j)*em(k)
c                        if(model.lt.24)then
                        if(nuvr.ge.1)then
                           em2(k)=0
                        endif
                  endif
                enddo
c         Recompute dm  because predicted mag changed 
c         Subtract the flux which is expected from the emi. lines to the observed flux
                avmago = 0. 
                avmagt = 0. 
                nf = 0 
                do k = 1,imagm
                 if (ab(k).gt.0.and.sab(k).gt.0.and.bused(k).eq.1)then
                      nf = nf + 1
                      if(buscal(k).eq.1)then
                         avmago = avmago + 
     >                   ((ab(k)-em2(k))*mag(k))/sab(k)**2
                         avmagt = avmagt + 
     >                   mag(k)*mag(k)/sab(k)**2
                      endif
                   endif
                enddo
                dm = avmago/avmagt
                chi2 = 0.
                do k = 1, imagm
                    if (bused(k).eq.1  .and. sab(k).gt.0) then
                       chi2 = chi2 + 
     >                 ((ab(k)-em2(k)-dm*mag(k))/sab(k))**2
                    endif
                enddo
   
                if(chi2.le.chiMinEm)then
                     fracMin=frac(j)
                     chiMinEm= chi2
                     dmmi=dm
               endif
            enddo
            do k = 1,imagm
               if(((dabs(flmoy(k)-(5007.*(1.+z))).le.flwidth(k)).or.
     .           (dabs(flmoy(k)-(4959.*(1.+z))).le.flwidth(k)).or.
     .           (dabs(flmoy(k)-(1216.*(1.+z))).le.flwidth(k)).or.
     .           (dabs(flmoy(k)-(4861.*(1.+z))).le.flwidth(k))))then
                   em(k)=fracMin*em(k)
                   if(nuvr.ge.1)then
                      em(k)=0
                   endif
               endif
            enddo
            dm=dmmi
          else
            do k = 1,imagm
              em(k)=0
            enddo
          endif
         endif
ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Chi2 weighted by Lir if lirprior used
         if (nlib.eq.1 .and. lirprior(1:1).eq.'Y') then
         if (dlirinp.gt.0 .and.dlirinp.le.2 .and. 
     >        lirinp.gt.5.and.lirinp.le.15.and.dm.gt.0)then
            lirpweight=-1.d0/(2*(dlirinp**2+0.2**2))
            lirpweight=lirpweight*(ldust+dlog10(dm)-lirinp)**2
            lirpweight=dexp(lirpweight)
c            lirpweight=lirpweight/(dlirinp*dsqrt(2*3.14159))
         endif   
         endif 
c         
ccccccccccccccccccccccccccccccccccccccccc
c Measurement of chi^2
         chi2 = 0.
         do k = 1, imagm
            chi2_fl(k)=-99.9 
            if (bused(k).eq.1  .and. sab(k).gt.0) then
               if(nlib.eq.1.and.addem(1:1).eq.'Y'.and.nuvr.le.4)then
                  chi2=chi2+((ab(k)-em(k)-dm*mag(k))/sab(k))**2
               else
                  chi2=chi2+((ab(k)-dm*mag(k))/sab(k))**2
               endif 
               chi2_fl(k)=((ab(k)-dm*mag(k))/sab(k))**2
            endif
         enddo
c         write(UO,*) model,z,extil,exti,chi2
c         if (chi2.lt.chimin(nlib) )
c     >  write(UO,*) 'best',model,z,extil,exti,chi2  
c  prior 
         if (nlib.eq.1) then 
           if (bp.gt.0.and.pweight.gt.0.d0) then            
c              if(pweight.le.0.d0)then 
c                chi2 = 1.d10
c              else         
             chi2 = chi2 - 2*dlog(pweight)
             if(chi2.lt.0.d0)then
               chi2=0.d0
             endif
c              endif 
           endif
           if (lirprior(1:1).eq.'Y'.and.lirpweight.gt.0)then
                chi2 = chi2 - 2*dlog(lirpweight)
           endif
         endif  
          
c
cccccccccccccccccccccccccccccccccccccccccc
c Extimate of the best chi at any redshift for the 3 types of objects
         if (chi2.lt.chimin(nlib) ) then 
            if(nlib.eq.1)then
               do k=1,imagm
                  chi2_fl_min(k)=chi2_fl(k)
               enddo
               if(addem(1:1).eq.'Y')then
                  do k=1,imagm
                    emMin(k)=em(k)
                  enddo
               endif
               reclphysb    = reclib(j)
c
               do k = 1,imagm
                  magb(k)=mag(k)*dm
               enddo
            endif
c
            chimin(nlib)  = chi2
            zmin(nlib)    = z
            dmmin(nlib)   = dm
            recmin(nlib)  = i
            reclmin(nlib) = recl
            extilmin(nlib)= extil
            recmin0(nlib) = irecz0  ! to keep magnitude at z=0
            imasmin(nlib) = model
            agemin(nlib)  = age
            extmin(nlib)  = exti
            ldustmin(nlib)= ldust
            zfmin(nlib)   = zfmod
            mag_abs(nlib) = abs_mag
         endif
ccccccccccccccccccccccccccccccccccccccccc
c study of the global chi2 evolution for gal library only 
         if (chi2.lt.1.e9.and.nlib.eq.1) then
           do nchi = 1,chimax
              if (DABS(z-chi(1,nchi)).le.1.d-5) then
                 chibay(nchi) = chibay(nchi)+dexp(-0.5*chi2)
                  if (chi2.lt.chi(2,nchi)) then 
                     chi(2,nchi) = chi2
                     recb(nchi)  = i
                     recb0(nchi) = irecz0  ! to keep magnitude at z=0
                     reclb(nchi) = recl
                     extilb(nchi)= extil
                     imasb(nchi) = model
                     extb(nchi)  = exti
                     ldustb(nchi)= ldust+dlog10(dm)
                     do k = 1,imagm
                        kcorb(k,nchi)=kcor(k) 
                     enddo
                     zfb(nchi)   = zfmod
                     ageb(nchi)  = age
                     zb(nchi)    = z
                     dmb(nchi)   = dm
                     mag_absb(nchi)= abs_mag 
                 endif
                  goto 2
              endif 
           enddo
         endif
 2       continue    
cccc  save in zchipara the dm, chi2
         if (nlib .eq.1 .and. dm .gt. 0 ) then 
           zchipara(1,i) = z
           zchipara(2,i) = age
           zchipara(3,i) = DBLE(extil)
           zchipara(4,i) = exti
           zchipara(5,i) = ldust  +dlog10(dm)
           zchipara(6,i) = physpara(2,recl)+dlog10(3.d18*400/2300**2)
           zchipara(6,i) = zchipara(6,i)-dlog10(Lsol)+dlog10(dm)
           zchipara(7,i) = physpara(3,recl)+dlog10(3.d18*1000/6000**2)
           zchipara(7,i) = zchipara(7,i)-dlog10(Lsol)+dlog10(dm)
           zchipara(8,i) = physpara(4,recl)+dlog10(3.d18*2000/22000**2)
           zchipara(8,i) = zchipara(8,i)-dlog10(Lsol)+dlog10(dm)
           zchipara(9,i) = physpara(5,recl)+dlog10(dm) 
           if (physpara(6,recl).gt.0) 
     >          zchipara(10,i)= dlog10(physpara(6,recl))+dlog10(dm)  
           if (physpara(7,recl).gt.0) 
     >          zchipara(11,i)= dlog10(physpara(7,recl))+dlog10(dm)  
           zchipara(12,i)= chi2 
         endif
c
      enddo                 ! END of LOOP ON LIBRARY 
cc    END   LOOP on LIBRARY FOR Chi^2 measurement 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Remove one band if large contributin to global Chi2
c      chi2 per filter > x% (levelChi2/100  = 60)
c     (Fobs-Fthe)/ErrF > x sigma (levelMag  = 5)
      sum_chi2=0.d0
      do k=1,imagm
        if (bused(k).eq.1  .and. sab(k).gt.0) then
         sum_chi2=sum_chi2+chi2_fl_min(k)
        endif
      enddo
      if (redofit.eq.0) then 
        do k=1,imagm
          if(bused(k).eq.1.and.ab(k).gt.0.and.sab(k).gt.0
     >     .AND.(chi2_fl_min(k)/sum_chi2).gt.levelChi2/100.
     >     .AND.(dabs(ab(k)-mag(k))/sab(k)).gt.levelMag) then
            bused(k)=0
            redofit=1
            goto 35 
          endif
        enddo
      endif
c      
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   Analysis of ML function.  Normalizing to peak at ML=1
      ndz=1
      if (chimin(1).ge.1e9 .OR. (zfix(1:1).eq.'Y' 
     >    .and. zs.ge.0 .and. zs.lt.zmax)  )  goto 4 
c      write(*,*) 'Chimin',chimin(1)
      if (chimin(1).lt.1e9) then
        do k = 1,chimax
          xp(k) = chi(1,k)
          yp(k) = dexp(-0.5*(chi(2,k) - chimin(1)))
          if (yp(k).le.1.d-30) yp(k) = 0.d0
        enddo
cccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   Redshift  uncertainties (zmin,zmax) for dChi2=1,2.71,6.63
        chibest=chimin(1)
        dchi = 1.0
        call DCHI2(chi,chimax,chibest,dchi,z68i,z68s)
        dchi = 2.71
        call DCHI2(chi,chimax,chibest,dchi,z90i,z90s)
        dchi = 6.63
c        dchi = 9.0
        call DCHI2(chi,chimax,chibest,dchi,z99i,z99s)
cccccccccccccccccccccccccccccccccccccccccccccccccccccc       
c   Extimates of the secondary peak in z
        call ZPEAK_SCALE(chi,yp,chimax,zstep,dz_win,min_thres,nb,ndz)
cccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   normalizing Pdz curve to 1
c   normalizing baysian zphot curve to 1
        z0=0.
        call TRAPZD(xp,chibay,chimax,z0,zmax,barea)
        if (barea.gt.0) then 
          call PROBAZBAY(xp,chibay,chimax,barea,zmed,zpdzi,zpdzs)
          zbay = zmed
          zbayi=zpdzi(2)     
          zbays=zpdzs(2)     
        endif
c   Redshift uncertainties (zmin,zmax) for 68%,90%,99%
        z0=0.
        call TRAPZD(xp,yp,chimax,z0,zmax,mlarea) 
        if (mlarea.gt.0) then 
c          call PROBAZ(xp,yp,chimax,mlarea,zpdzi,zpdzs,probz)
c          zml68i=zpdzi(1)
c          zml68s=zpdzs(1)
c          zml90i=zpdzi(2)
c          zml90s=zpdzs(2)
c          zml99i=zpdzi(3)
c          zml99s=zpdzs(3)
          call PROBAZBAY(xp,yp,chimax,mlarea,zmed,zpdzi,zpdzs)
          zml68i=zpdzi(2)
          zml68s=zpdzs(2)
          zml90i=zpdzi(3)
          zml90s=zpdzs(3)
          zml99i=zpdzi(4)
          zml99s=zpdzs(4)
c
          do k = 1, chimax
            yp(k) = yp(k) / mlarea
          enddo
        endif
cccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   Computing fraction included between +/- Dz = 0.1*(1+z)
        dzp=0.1
        dzml = dzp*(1+zmin(1))
        zinf = zmin(1) - dzml
        zsup = zmin(1) + dzml
        if (zinf.lt.0)    zinf = 0
        if (zsup.gt.zmax) zsup = zmax
        call TRAPZD(xp,yp,chimax,zinf,zsup,summl)
        pdz(1) = summl*100.  
        if (ndz.gt.1) then 
          do k = 2 , ndz
             dzml = dzp*(1+zb(nb(k)))
             zinf = zb(nb(k)) - dzml
             zsup = zb(nb(k)) + dzml
             if (zinf.lt.0)    zinf = 0
             if (zsup.gt.zmax) zsup = zmax
             call TRAPZD(xp,yp,chimax,zinf,zsup,summl)
             pdz(k) = summl*100.                  
          enddo
        endif
ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  Computes proba in various intervals of z
        if (npdz.gt.1) then
           j = 1
           do k = 1, npdz, 2
              zinf=int_pdz(k)
              zsup=int_pdz(k+1)
              call TRAPZD(xp,yp,chimax,zinf,zsup,summl)
              dzpdz(j)=summl*100
              j = j + 1
           enddo
        endif    
        do k = 1, chimax
          yp(k) = yp(k) * mlarea
        enddo
      else
         pdz(1) = -99
        if (npdz.gt.1) then
           j = 1
           do k = 1, npdz, 2
              if (int_pdz(k).le.zmin(1).and.
     >             int_pdz(k+1).gt.zmin(1)) then
                 dzpdz(j) = 100
              else
                  dzpdz(j) = 0
              endif
              j=j+1
           enddo
        endif    
      endif
ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
 4    if ( fastmod(1:1).eq."Y".AND.chimin(1).gt.1e8 .AND. 
     >    redoing.eq.0 ) then
         redoing=1
         goto 36
      endif  
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c    REDSHIFT INTERPOLATION  if  ZINTP=Y or ZFIX=Y
      if (zmin(1).ge.0 .and.reclmin(1).ge.1) then
c   Z_best replaced by Z_spec if defined 
         if(zfix(1:1).eq.'Y'.AND.zs.ge.0.and.zs.lt.zmax)then 
             zbest=zmin(1)
             mod_dist2 = funz(zbest,h0,om0,l0)
             zmin(1)=zs
             mod_distb(1) = funz(zs,h0,om0,l0)
             dmcor=10**(0.4*(mod_distb(1)-mod_dist2))
c       write(*,*) spec,zbest,mod_dist2,zs,dmmin(1),dmcor,"  "
             dmmin(1)=dmmin(1)*dmcor
c  Interpolation of Z_best (zmin(1)) via Chi2 curves
         elseif (zintp(1:1).eq.'Y') then
             zbest=zmin(1)
             mod_dist2 = funz(zbest,h0,om0,l0)
             call int_parab(chi,chisize,chimax,zbest,zintb)
             zmin(1)=zintb
             mod_distb(1) = funz(zintb,h0,om0,l0)
             dmcor=10**(0.4*(mod_distb(1)-mod_dist2))
             dmmin(1)=dmmin(1)*dmcor 
         else
             zbest=zmin(1)
             mod_distb(1) = funz(zbest,h0,om0,l0)
             dmcor=1.d0
         endif
c         write(*,*) 'zmin and zspec ',zmin(1),zs
c         write(*,'(A,1x,f6.4,1x,I8,1x,I3)')  zfix(1:1),zmax,model,nlib
ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   Best Chi^2 for the physical quantities included  
c   in the orignal (.phys) file library
c  Age (yr)  : same as agemin(1) indeed !
         if (physpara(1,reclmin(1)).gt.0) 
     >       physpbest(1)=(physpara(1,reclmin(1))) 
c  Log Luv (Lo)
         if (physpara(2,reclmin(1)).gt.-90) then 
           physpbest(2)=physpara(2,reclmin(1))+dlog10(3.d18*400/2300**2)
           physpbest(2)=physpbest(2)-dlog10(Lsol)+dlog10(dmmin(1))
         endif
c  Log Lr  (Lo) 
         if (physpara(3,reclmin(1)).gt.-90) then
          physpbest(3)=physpara(3,reclmin(1))+dlog10(3.d18*1000/6000**2)
          physpbest(3)=physpbest(3)-dlog10(Lsol)+dlog10(dmmin(1))
         endif
c  Log Lk  (Lo)
         if (physpara(4,reclmin(1)).gt.-90) then
         physpbest(4)=physpara(4,reclmin(1))+dlog10(3.d18*2000/22000**2)
           physpbest(4)=physpbest(4)-dlog10(Lsol)+dlog10(dmmin(1))
         endif 
c  Log Ldust (Lo)
         if (physpara(5,reclmin(1)).gt.-90) then
         physpbest(5)=physpara(5,reclmin(1))+dlog10(dmmin(1))
         endif 
c  Log Mass (Mo)
         physpbest(6)=-99.d0
         if (physpara(6,reclmin(1)).gt.0) 
     >   physpbest(6)=dlog10(physpara(6,reclmin(1)))+dlog10(dmmin(1))
c  Log SFR (Mo/yr)
         physpbest(7)=-99.d0
         if (physpara(7,reclmin(1)).gt.0) 
     >   physpbest(7)=dlog10(physpara(7,reclmin(1)))+dlog10(dmmin(1))
c
      endif
cccccccccccccccccccccccccccccc
      do k = 1, 6
         parainf(k) =-99
         parasup(k) =-99
         paramed(k) =-99.99
      enddo
c      write(*,*) 
c      write(*,*) 'Z',zmin(1) , z68i,z68s
c      write(*,*) 'best:',physpbest(1),physpbest(5),physpbest(6),
c     >     physpbest(7),physpbest(7)-physpbest(6)
      if ((z68i.ge.0 .and. z68s.gt.z68i) .OR. ZFIX(1:1).eq.'Y' ) then
         if (z68i.ge.0 .and.z68s.gt.z68i) then 
            zinf = z68i
            zsup = z68s
         endif
        if (ZFIX(1:1) .eq. 'Y') then 
           zinf = zmin(1)-zstep
           zsup = zmin(1)+zstep
        endif
c        write(*,'(A,6(f8.3,2x))') 'IN',zmin(1),zinf,zsup,zstep,z68i,z68s
        call chi_para(zchipara,liblength,zinf,zsup,zstep,parainf,
     >   parasup,paramed)
      endif 
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
c Chi^2 on the FIR luminosity based on the best redshift 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
c  recompute the FIR context  
      nbufir=0
      nbsfir=0
      do k = 1, imagm
        busfir(k)=0
        if (fir_cont.le.0) then
         if ((flmoy(k)/10000./(1+zmin(1))).ge.fir_lmin) busfir(k)=1
        else
           busfir(k)=bdincl(k-1,fir_cont,imagm-1)
         if ((flmoy(k)/10000./(1+zmin(1))).lt.fir_lmin) busfir(k)=0
        endif
        if (busfir(k).eq.1) nbufir=nbufir+1
        if (fir_scale.le.0) then      
          if ((flmoy(k)/10000./(1+zmin(1))).ge.fir_lmin
     >          .and.busfir(k).eq.1) bscfir(k)=1
        else
          bscfir(k)=bdincl(k-1,fir_scale,imagm-1)
          if (busfir(k).eq.0) bscfir(k)=0
        endif
        if (bscfir(k).eq.1) nbsfir=nbsfir+1
      enddo 
c
      irecfirz0=-1
      irecfirb =-1
      reclfirb =-1
      libfirb  =-1    
      modfirb  =-1
      lumfirb  =-1
      lirmed   =-1    
      lirinf   =-1    
      lirsup   =-1   
      nf_fir   =-1
      cont_fir =-1
      dmfirb   =-1
      chi2_fir =1.e9 
      dmag     =1.e19
      dmcor=1.
      do k = 1, nlibext
         dmagir(k) = 1.e19
         lumlirb(k) = -99
      enddo
      do k = 1,imagm
        absfir(k) = -99. 
        absfiro(k) = -99.
        kcfirb(k) = 0. 
        magfirb(k) = -99.
        kccor(k)=1.
      enddo
      do j = 1,nfir
        lumtemp(j) = -99.99
        dmirtemp(j) = 99.99
      enddo
c
      if (nlibext.ge.1 .and. zmin(1).ge.0) then
         lirmax=225    
         dlir=0.04
         do k = 1,lirmax
           lumir(k) = 6.0+ dlir*(k-0.5)
           chilir(k)= 0.
         enddo
         xmin=lumir(1)
         xmax=lumir(lirmax)
c
         dmir    = 0
         reclfirb=-1
         chirmin =1.e9
         chi2_fir=0
ccccccccccccccccccccccccccccccccccccccccccccccccc
c   SUBTRACT THE STELLAR SED COMPONENT TO THE OBSERVED FIR FLUXES.
c   CHECH NUMBER OF BANDS USED with Lbda/(1+z)>Lbmin um
         cont_fir=0  
         nf_fir=0  
         do k  = 1, imagm 
            abc(k)  = ab(k)
            sabc(k) = sab(k)
            if (substar(1:lnblnk(substar)).eq.'YES'.and. busfir(k).eq.1
     >          .and.magb(k).gt.0
     >          .and.(flmoy(k)/10000./(1+zmin(1))).le.25.0) then 
              if(sab(k).gt.0 .and. 
     >           (ab(k)-2*sab(k)).gt.magb(k)) then 
c     >           (ab(k)-magb(k)-1*sab(k)).gt.0) then 
                 abc(k)  = ab(k)-magb(k)
                 sabc(k) = dsqrt(sab(k)**2+magb(k)**2)
              elseif (sab(k).gt.0 .and. 
     >           (ab(k)-2*sab(k)).le.magb(k)) then 
c              do not subtract stellar contribution but enlarge
c              the uncertainties by possible stellar contribution         
                 abc(k)  = ab(k)
                 sabc(k) = dsqrt(sab(k)**2+magb(k)**2)
              else
                 busfir(k)=0
                 bscfir(k)=0
              endif
            endif 
            if (abc(k).gt.0.and.sabc(k).gt.0.and.busfir(k).eq.1
     >        .and. flmoy(k)/10000./(1+zmin(1)) .ge. fir_lmin ) then
                nf_fir=nf_fir+1
                cont_fir=cont_fir +2**(k-1)
            endif
         enddo
c         write(*,*) "  "  
        if (nf_fir.eq.0) write(UO,*) 'no appropriated FIR bands->OFF' ! stop the FIR analysis
         if (nf_fir.eq.0) goto 34    ! stop the FIR analysis
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   START THE LOOP 
         do j = 1,nfir    
            kbestir(liblfir(j))=-1
         enddo
         kbest=-1
         do j = 1, nfir
           if (zlfir(j).lt.(zstep/10.d0))  irecfirz0=j
ccccccccccccccc  IF   SEVERAL BANDS USED ccccccccccccccccccccccccccccccccc
           if (SNGL(dabs(zlfir(j)-zmin(1))).le.(SNGL(zstep/2.d0))
     >         .AND.nf_fir.gt.1)  then 
c            TAKE OUT THE FLUX CORRECTION  DUE TO DISTANCE MODULUS:
c            Zlib ne Zspec in the Chi2 analysis 
c  DMcor :
             ztemp=zlfir(j)
             mod_dist2 = funz(ztemp,h0,om0,l0)
             dmcor=10**(0.4*(mod_dist2-mod_distb(1)))
c Kccor : 
             sens=0 
             if (zmin(1).gt.zlfir(j)) sens=+1
             if (zmin(1).lt.zlfir(j)) sens=-1
             akir=0
             if (sens.ne.0)
     >         akir=dabs((zlfir(j)-zmin(1))/(zlfir(j)-zlfir(j+sens)))
             do k = 1, imagm
c               if (sabc(k).gt.0.and.busfir(k).eq.1) then 
                  if(modlfir(j).ne.modlfir(j+sens))  akir=0.d0
                  if(klfir(k,j+sens).gt.90)          akir=0.d0
                  kzobs(k)=akir*klfir(k,j+sens)+(1-akir)*klfir(k,j)
                  kccor(k)=10**(0.4*(klfir(k,j)-kzobs(k)))
c               endif 
             enddo
c  Chi^2  : scaling 
             nf    = 0 
             dmir  = 0.
             avmagt= 0. 
             avmago= 0. 
             do k = 1,imagm
               if (abc(k).gt.0.and.sabc(k).gt.0.and.bscfir(k).eq.1)then
                 nf = nf + 1
                 avmago=avmago+
     >            abc(k)*maglfir(k,j)*dmcor*kccor(k)/sabc(k)**2
                 avmagt=avmagt+
     >        maglfir(k,j)*maglfir(k,j)*dmcor**2*kccor(k)**2/sabc(k)**2
               endif
             enddo
             if (nf.ge.1 .and. fir_frsc(1:1).eq.'Y') then
cc Scaling only account for the real flux rescaling not the difference in Distance Modulus 
               dmir = avmago/avmagt
c               write(*,*) 'zsp zmod model , dmir=',
c     >                 zmin(1),zlfir(j),modlfir(j),dmir
             else
               dmir=1.d0 
             endif
cc
             chi2_fir = 0.d0
             nf_fir=0
             do k = 1, imagm
                if (busfir(k).eq.1 .and. sabc(k).gt.0 ) then
       chi2_fir=chi2_fir+
     >      ((abc(k)-dmir*maglfir(k,j)*dmcor*kccor(k))/sabc(k))**2
                     nf_fir=nf_fir+1
                endif 
             enddo
c
             if (chi2_fir.lt.chirmin ) then
                chirmin = chi2_fir 
                zfirb   = zmin(1)
                dmfirb  = dmir
                modfirb = modlfir(j)                   
                libfirb = liblfir(j)
                reclfirb= reclfir(j)
                lumfirb = lumlfir(j)+dlog10(dmfirb)
                irecfirb= irecfirz0 
                do k = 1, imagm
                  kcfirb(k)=kzobs(k)
                enddo
c kcor interpolated   
c                sens=idnint(dsign(1.d0,(zmin(1)-zlfir(j))))
c                akir=dabs((zlfir(j)-zmin(1))/(zlfir(j)-zlfir(j+sens)))
c                   if(modlfir(j).ne.modlfir(j+sens))  akir=0.d0
c                   if(klfir(k,j+sens).gt.90)          akir=0.d0
c                   kcfirb(k)=akir*klfir(k,j+sens)+(1-akir)*klfir(k,j)
c                enddo
c best magnitude 
                do k = 1,imagm 
                  if (maglfir(k,j).gt.0)
     >magfirb(k)=-2.5*dlog10(dmir*maglfir(k,j)*dmcor*kccor(k))-48.59
                enddo 
c
             endif
c
             if (chi2_fir.lt.1.e9) then
                  do nchi = 1,lirmax
c           if ((lumlfir(j)+dlog10(dmir)).ge.(lumir(nchi)-dlir/2.d0)
c     >   .and.(lumlfir(j)+dlog10(dmir)).lt.(lumir(nchi)+dlir/2.d0))then
           if ( SNGL(DABS(lumlfir(j)+dlog10(dmir)-lumir(nchi)))
     >                 .le. SNGL(dlir/2.d0) )then                       
                       chilir(nchi)=chilir(nchi)+dexp(-0.5*chi2_fir)
                    endif
                  enddo
             endif 
           endif
ccccccccccccccc  IF  ONLY ONE FIR  BAND USED ccccccccccccccccccccccccccccccccc
c          choose the closest app-magitude in that filter.  
           if (SNGL(dabs(zlfir(j)-zmin(1))).le.(SNGL(zstep/1.d0))
     >          .AND.nf_fir.eq.1)then 
c            TAKE OUT THE FLUX CORRECTION  DUE TO DISTANCE MODULUS AND KCOR:
c            Zlib ne Zspec in the Chi2 analysis 
c  m(zlib)=M+DM(zlib)+kc(zlib)  &   m(zobs)=M+DM(zobs)+kc(zobs)
c  m(zlib)-m(zobs)= (DM(zlib-DM(zobs)) +(kc(zlib)-kc(obs))
c  f(zobs)/f(zlib)= 10^0.4[DM(zlib)-DM(zobs)] . 10^0.4[kc(zlib)-kc(obs)]         
c  Correction  f(zlib->zobs) = fzlib * dm_cor * kcor_cor
c Dist modulus :
             ztemp=zlfir(j)
             mod_dist2 = funz(ztemp,h0,om0,l0)
             dmcor=10**(0.4*(mod_dist2-mod_distb(1)))
c Kcor : 
             sens=0 
             if (zmin(1).gt.zlfir(j)) sens=+1
             if (zmin(1).lt.zlfir(j)) sens=-1
             akir=0
             if (sens.ne.0)
     >         akir=dabs((zlfir(j)-zmin(1))/(zlfir(j)-zlfir(j+sens)))
             do k = 1, imagm
c               if (sabc(k).gt.0.and.busfir(k).eq.1) then 
                  if(modlfir(j).ne.modlfir(j+sens))  akir=0.d0
                  if(klfir(k,j+sens).gt.90)          akir=0.d0
                  kzobs(k)=akir*klfir(k,j+sens)+(1-akir)*klfir(k,j)
                  kccor(k)=10**(0.4*(klfir(k,j)-kzobs(k)))
c               endif 
             enddo
c  test no correction 
c             do k = 1,imagm
c               kccor(k) = 1
c             enddo 
c             dmcor=1
c  Best models 
             kmin=0
             do k = 1, imagm
               if (sabc(k).gt.0.and.busfir(k).eq.1) then
               lumtemp(j)=lumlfir(j)+
     >                  dlog10(abc(k)/maglfir(k,j)/dmcor/kccor(k)) 
               dmirtemp(j)=dabs(2.5*dlog10(abc(k)/maglfir(k,j)))
           if (DABS(dlog10(abc(k)/maglfir(k,j)/dmcor/kccor(k))).le.
     >                                  DABS(dlog10(dmag)))then
                   dmag=  (abc(k)/maglfir(k,j)/dmcor/kccor(k))
                   kbest=j
                   kmin=1
                 endif 
           if (DABS(dlog10(abc(k)/maglfir(k,j)/dmcor/kccor(k))).le.
     >                      DABS(dlog10(dmagir(liblfir(j)))))then
             dmagir(liblfir(j))=(abc(k)/maglfir(k,j)/dmcor/kccor(k))
                   kbestir(liblfir(j))=j
                 endif 
               endif
             enddo
             if (kmin.eq.1) then   
                zfirb    = zmin(1)                
                dmfirb   = dmag
                modfirb  = modlfir(kbest)                   
                libfirb  = liblfir(kbest)
                reclfirb = reclfir(kbest)
                lumfirb  = lumlfir(kbest)+dlog10(dmfirb)
                irecfirb = irecfirz0 
                do k = 1, imagm
                   kcfirb(k)=kzobs(k)
                enddo 
c kcor interpolated   
c                sens=idnint(dsign(1.d0,(zmin(1)-zlfir(j))))
c                akir=dabs((zlfir(j)-zmin(1))/(zlfir(j)-zlfir(j+sens)))
c                do k = 1, imagm
c                   if(modlfir(j).ne.modlfir(j+sens))  akir=0.d0
c                   if(klfir(k,j+sens).gt.90)          akir=0.d0
c                   kcfirb(k)=akir*klfir(k,j+sens)+(1-akir)*klfir(k,j)
c                enddo
c best magnitude 
                do k = 1,imagm 
                  if (maglfir(k,j).gt.0)
     >magfirb(k)=-2.5*dlog10(dmag*maglfir(k,j)*dmcor*kccor(k))-48.59
                enddo 
             endif
             if (kbestir(liblfir(j)).gt. 0) then
               lumlirb(liblfir(j))= lumlfir(kbestir(liblfir(j)))
     >                              + dlog10(dmagir(liblfir(j)))
c              write(*,*) liblfir(j),dmagir(liblfir(j))
             endif   
           endif
         enddo
ccccccccccccccc    COMPUTE THE MEDIAN   ccccccccccccccccccc
         if (nf_fir.gt.1) then 
c            compute Ltir from the median of the Sum(exp(-0.5xchi2)) distribution  
            call TRAPZD(lumir,chilir,lirmax,xmin,xmax,barea)
            if (barea.gt.0) then 
              call PROBAZBAY(lumir,chilir,lirmax,barea,zmed,zpdzi,zpdzs)
               lirmed = zmed     
               lirinf =zpdzi(2)  
               lirsup =zpdzs(2)  
            endif
         else
           lirmed=0
           do k = 1,nlibext
             lirmed=lirmed +  lumlirb(k)
           enddo
           lirmed = lirmed/nlibext
           lirinf =0
           lirsup =0
           if (nlibext.gt.1) then 
             do k = 1,nlibext 
               lirinf =  lirinf + (lumlirb(k)-lirmed)**2
             enddo
             lirsup = lirmed+dsqrt(lirinf/(nlibext-1))/2.
             lirinf = lirmed-dsqrt(lirinf/(nlibext-1))/2.
           else
             if (kbest.gt.0) then 
c  only 1 library: select the mean between Lir_up and Lir_down
c                 rather than the best value
               ntempir=0
               lirmed=0
               do j=1,nfir
                 ltemp(j)=0
               enddo
c               write(*,*) "  "
               do j = 1,nfir     
c    modfirb  = modlfir(kbest)
c    zfirb    = zmin(1)                
c    lumfirb  = lumlfir(kbest)+dlog10(dmfirb)
c       
                 if (lumtemp(j).gt.0) then 
                 if (modlfir(j).ge.(modfirb-1) .and.
     >               modlfir(j).le.(modfirb+1) ) then
c        write(*,*) "1:", ntempir,kbest,j,modlfir(j),modfirb,
c     > lumlfir(kbest),lumfirb,lumlfir(j),lumtemp(j),zlfir(j)
 
                 if ( (lumfirb.le.lumlfir(kbest).and.
     >               lumfirb.ge.lumlfir(j)) .OR.
     >              (lumfirb.ge.lumlfir(kbest).and.
     >               lumfirb.le.lumlfir(j)) .OR.
     >     DABS(lumlfir(j)-lumlfir(kbest)).le.0.01 ) then 
                    lirmed=lirmed+lumtemp(j)
                    ntempir=ntempir+1       
                    ltemp(ntempir)=lumtemp(j)
c        write(*,*) "2:",ntempir,kbest,j,modlfir(j),modfirb,
c     > lumlfir(kbest),lumfirb,lumlfir(j),lumtemp(j),zlfir(j)
                 endif
                 endif
                 endif
               enddo
               if (ntempir.gt.0) then 
                if (ntempir.ne.4) then
                  write(*,*) " " 
              write(*,*) "Not equal 4  ",ntempir,zlfir(kbest)
                  write(*,*) " " 
                endif
                 lirmed=lirmed/ntempir
                 if (ntempir.ge.2) then
                   lmini=999.99
                   lmaxi=-999.99
                   do j=1,ntempir
                      if (ltemp(j).gt.lmaxi) lmaxi=ltemp(j)
                      if (ltemp(j).lt.lmini) lmini=ltemp(j)
                   enddo  
                   lirsup= lirmed+DABS(lmaxi-lmini)/2.
                   lirinf= lirmed-DABS(lmaxi-lmini)/2.
                 endif
               endif
             endif
           endif
c           write(*,'(8(f8.3,2x))') 
c     >          (lumlirb(k),k=1,nlibext),lirmed,lirinf,lirsup
         endif
ccccccccccccccc   EXTRACT THE ABSOLUTE LUMINOSITY   cccccccccccccc
         if (dmfirb.gt.0 .and. irecfirb.gt.0) then 
           do k = 1,imagm 
c    from the best models 
             if (maglfir(k,irecfirb).gt.0.) then
               absfir(k) = -2.5*dlog10(maglfir(k,irecfirb))
     >                     -48.59-2.5*dlog10(dmfirb)
             endif
c    from the observed band flux 

           enddo
         endif
      endif
 34   continue
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
c END   Chi^2 on the  FIR luminosiaty based on the best redshift 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
c Chi^2 on the  Phys parameters based on the LIB_PHYS  library and best z
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
c   
      do j = 1,50
        ppbest(j)  =-99.d0    
        ppmed(j)   =-99.d0    
        ppinf(j)   =-99.d0    
        ppsup(j)   =-99.d0
      enddo   
      do k = 1, imagm
        fluxphys(k)=-99
      enddo
      chipbest=-99.d0
      reclpbest=-99  
      if(nlibphys.eq.1.and.
     >   (zmin(1).ge.0 .OR.(zfix(1:1).eq.'Y'.and.zs.ge.0))) then
         if (zmin(1).gt.0 )                zbest=zmin(1)
         if (zfix(1:1).eq.'Y'.and.zs.ge.0) zbest=zs
         if (zbest.le.zrecp(nzrecp)) then 
          call chi_libphys(libphys,libppara,
     >    zrecp,zrecpi,zrecps,nzrecp,phys_max,
     >    zbest,ab,sab,bused,buscal,lirmed,
     >    ppbest,chipbest,reclpbest,dmpbest,ppmed,ppinf,ppsup,nppara,
     >    fluxphys,kcorphys,magphys0)
          do k = 1, imagm
            if (fluxphys(k).gt.0) then 
              fluxphys(k)= -2.5*dlog10(fluxphys(k))-48.59    
            else
              fluxphys(k)=-99
            endif
          enddo
         endif
      endif
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c END   Chi^2 on the  Phys library 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  writing output   (correspondance with goto 4)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      do k = 1, imagm
         if ( ab(k).gt.0 .and. sab(k).gt.0 ) then 
           sab(k) = 1.086*sab(k)/ab(k)
           if (magtyp(1:1).eq.'A') then 
              ab(k)=-2.5*dlog10(ab(k))-48.59
           elseif (magtyp(1:1).eq.'V') then 
              ab(k)=-2.5*dlog10(ab(k))+zp(k)
           endif   
         elseif (ab(k).gt.0 .and. sab(k).le.0 ) then
             sab(k) = -1
             if (magtyp(1:1).eq.'A') then
                ab(k)=-2.5*dlog10(ab(k))-48.59
             elseif (magtyp(1:1).eq.'V') then
                ab(k)=-2.5*dlog10(ab(k))+zp(k)
             endif   
         elseif (ab(k).le.0 ) then
            sab(k) = 99.
            ab(k) = 99.
         endif
      enddo
cccccccccccccccccccccccccccccccccccccccccc
c  Estimation of the kappa correction kap(mod) = mabs(z) - mabs(0)
c        kap = mapp(z) - fz(z) - (mapp(0) -fz(0))  
      if (zmin(1).ge.0) then
         do k = 1, imagm
            if (fastmod(1:1).eq."Y") then
               magm0(k) = maglib(k,reclist(recmin0(1)))
             else    
               magm0(k) = maglib(k,recmin0(1))
             endif 
         enddo     
c   Interpolation for k-correction  and theoretical magnitudes if needed
         if(fastmod(1:1).eq."Y" )then
             index=reclist(recmin(1))
         else
             index=recmin(1)
         endif
         call kInterp(zmin(1),index,imagm,zlib,modlib,klib,kap)
         call kInterp(zmin(1),index,imagm,zlib,modlib,maglib,magm)
c  Computes  ABSOLUTE MAGNITUDES  FOR GALAXIES with various METHOD 
         call absMagPro(method,spec,imagm,bapp,zstep,dz,h0,om0,l0,
     >            fobs,fobs4,mbused,zmin(1),dmmin(1),ab,magm,sab,kap,
     >            magm0,minkcol,goodfilter,minkcolor,mabs)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  compute Zmax  : zvmax(nzmax)
c  diff_mag=Mag_lim-mag_obs=(DM+kcor)(zmax)-(DM+kcor)(z) 
         if (nzmax.ge.1) then
c
           do k = 1, nzmax       
             zvmax(k)= -99.99  
             zband= zmfilt(k)
c             write(*,*) 
c             write(*,*) zband,zmin(1),mod_distb(1),kap(zband)
c             write(*,*) 
c             
             if (zband.lt.1 .or. zband.gt.imagm) goto 123
c           
             diff_mag=zmlim(k)-abo(zband)
             if (diff_mag.lt.0) goto 123   
             diff_mag=diff_mag+mod_distb(1)+kap(zband)
c
c  check the best index accroding to Z_interpolate
             if (zmin(1).gt.zlib(index)) index=index+1  
c
             call z_vmax(index,diff_mag,zband,modlib,extlib,ebvlib,
     >         agelib,klib,zlib,dmlib,liblength,zmaxlib)
             zvmax(k)=zmaxlib
 123         continue
c
           enddo
         endif 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
         do k = 1, imagm 
            magm(k) = magm(k) -2.5*dlog10(dmmin(1))
            if(addem(1:1).eq.'Y') 
     >  magm(k)=-2.5*dlog10(10**(-0.4*(magm(k)+48.59))+emMin(k))-48.59
         enddo   
      else                 ! if (Chi^2>1.e9 & no Zphot) 
         do k = 1, imagm
            kap(k)  = -99.99
            mabs(k) = -99.99
            magm(k) = -99.99
         enddo   
      endif
c
cccccccccccccccccccccccccccccccccccccccccc
c Mag_abs for QSO 
      if (zmin(2).ge.0) then 
         zbest=zmin(2)
         mod_distb(2) = funz(zbest,h0,om0,l0)
         do k = 1, imagm
            if (fastmod(1:1).eq."Y" .or.fastmod(1:1).eq."y" ) then
               kapq(k)   = klib(k,reclist(recmin(2)))
            else    
               kapq(k)   = klib(k,recmin(2))
            endif    
            mabsq(k)=ab(k)-mod_distb(2)-kapq(k)
         enddo              
      else
         do k = 1, imagm
            kapq(k)  = -99.99
            mabsq(k) = -99.99
         enddo   
      endif
c
cccccccccc    SELECT OBJECTS FOR  AUTO-ADAPT
      if (zmin(1).ge.0.d0 .AND. autoadapt(1:1).eq.'Y' .and.
     >    realise.le.1 ) then
            if(ngals_ada.lt.zadapt)then
              cont_ada(ngals_ada)= new_cont
              if(imasmin(1).lt.admmin .or.imasmin(1).gt.admmax) goto 1
              mod_ada(ngals_ada) = imasmin(1)
              zs_ada(ngals_ada)  = zs
              do k=1,imagm
                 ab_ada(ngals_ada,k)  = ab(k)
                 sab_ada(ngals_ada,k) = sab(k)*fac_err
                 magm_ada(ngals_ada,k)= magm(k)
              enddo
              ngals_ada = ngals_ada+1
            endif
      endif
c
c10001 format(i9,1x,f9.3,1x,i6,1x,E12.6,1x,f9.3,1x,70(f12.5,1x))
c
cccccccccccccccccccccccccccccccccccccccccccc
c objects measured 
      do k = 1,3 
         if (chimin(k).lt.1e9) nmeas(k)=nmeas(k) + 1
      enddo  
c      write(*,*) 'writing output'
cccccccccccccccccccccccccccccccccccccccccc
c        WRITING IN OUTPUT FILES 
c
      if(realise.ge.2 .OR. autoadapt(1:1).eq.'N')then
        call WRITE_OUT(wpara,iwpara,spec,
     >  zmin,chimin,imasmin,agemin,extilmin,extmin,zfmin,dmmin,
     >  pdz,mag_abs,new_cont,zs,str_inp,zb,
     >  chi,imasb,ageb,extb,zfb,dmb,mag_absb,nb,
     >  abo,sabo,kap,mabs,imagm,goodfilter,magm,
     >  dzpdz,npdz,
     >  z68i,z68s,z90i,z90s,z99i,z99s,
     >  zml68i,zml68s,zml90i,zml90s,zml99i,zml99s, 
     >  zbay,zbayi,zbays,
     >  mabsq,kapq, 
     >  nbus,nbul,
     >  mod_distb,
     >  parainf,parasup,paramed,
     >  lumfirb,libfirb,modfirb,chirmin,nf_fir,dmfirb,
     >  lirmed,lirinf,lirsup,absfir,kcfirb,magfirb,
     >  physpbest,
     >  ppbest,chipbest,reclpbest,
     >  ppmed,ppinf,ppsup,
     >  fluxphys,magphys0,kcorphys,
     >  zvmax,nzmax,
     >  str_out,iwout)
c
        write(4,'(1000(A,2x))') 
     >  (str_out(k)(1:lnblnk(str_out(k))),k=1,iwout)
c      write(*,*) 'writing output'
c
ccccccccccccccccccccccccccccccccccccccccc
c        writing PDZ files 
        if (outpdz(1:4).ne."NONE" .AND. fastmod(1:1).ne."Y") then
          do k = 1,chimax
             if (yp(k).le.1.e-10) then
                do i = 1, nfabs
                   pdz_mabs(i,k) =-99.99
                enddo
             else 
                pdz_z = chi(1,k)
                pdz_dm= dmb(k)
                pdz_distm=dist_mod(k)
                do i = 1,imagm 
                   pdz_kcor(i)= kcorb(i,k)
                   magm0(i)   = maglib(i,recb0(k))
                   magm(i)    = maglib(i,recb(k))
                enddo
c       Computes  ABSOLUTE MAGNITUDES  FOR GALAXIES with various METHOD 
                do i = 1,nfabs
                 fabs=pdz_fabs(i)
                 call absMag1F(method,spec,fabs,bapp,zstep,dz,
     >      pdz_distm,fobs,fobs4,mbused,pdz_z,pdz_dm,ab,magm,sab,
     >      pdz_kcor,magm0,minkcol,goodfilter,minkcolor,pdz_mabsz)
                 pdz_mabs(i,k) = pdz_mabsz
               enddo
             endif
          enddo
          write(44,'(500(f8.3,1x))') (yp(k),k=1,chimax)
          write(45,'(500(i6,1x))')   (imasb(k),k=1,chimax)
          do i = 1,nfabs
            ufabs=45+i 
            write(ufabs,'(500(f8.3,1x))') (pdz_mabs(i,k),k=1,chimax)
          enddo
        endif
c        write(UO,*) '  end writing ..'
      endif

c     END   WRITING  IN OUTPUT FILES  
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c info in screen
      if (UO .eq. 6) then    
      if(autoadapt(1:1).eq.'N' .OR. realise.ge.2 )then
         if (fastmod(1:1).eq."Y") then
            write(UO,603) spec,numrec,zmin(1),pdz(1),mag_abs(1),
     >                    zmin(2),mag_abs(2),imasmin(3),char(13)
            call flush(UO)
         else
           if (nlibext.ge.1) then  
              write(UO,601) spec,zmin(1),pdz(1),nf_fir,
     >   libfirb,modfirb,lumfirb,lirmed,lirinf,lirsup,char(13)
              call flush(UO)

           else
              write(UO,602) spec,zmin(1),pdz(1),mag_abs(1),
     >                    zmin(2),mag_abs(2),imasmin(3),char(13)
              call flush(UO)
            endif   
         endif   
      endif
      endif
c
 601  format("object ->",I10,1x,2(f6.2,1x),3(I4,1x),4(f8.3,1x),a1,$)
 602  format("object ->",I10,1x,5(f6.2,1x),I4,a1,$)
 603  format("object ->",2(I10,1x),5(f6.2,1x),I4,a1,$)
c    END   info on screen 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  writing the Chi2 parameters out 
      if (outchi(1:1).eq.'Y') then
         write(ospec,'(i9.9)') spec 
         ospec= 'Id' // ospec(1:lnblnk(ospec)) // '.chi'
         open(7,file=ospec,status='unknown')
c        1  2   3    4    5     6    7   8   9      10   11  12   
c        z,age,ext-l,ebv,ldust, luv, lr, lk, ldust, mo, sfr  chi2
         write(7,'(2(A,1x))') '# Z Age Extlaw EB_V Ldust',
     >                        '   Luv Lr Lk Ldust2 Mo SFR Chi2'
         do i = 1,liblength
           if (zchipara(1,i).ge.0) 
     >       write(7,'(12(E12.6,2x))') (zchipara(j,i),j=1,12)
         enddo
         close(7)
      endif
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Writing the spectra and chi2 curves  
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (outsp(1:1).eq.'Y') then
c output file name 
        write(ospec,'(i9.9)') spec 
        ospec= 'Id' // ospec(1:lnblnk(ospec)) // '.spec'
cccccccccccccccccccccccccccccccccc
c Lbda min - max 
        lbdminwr = 1.e20
        lbdmaxwr = 0 
        do i = 1, imagm 
          if ( (flmoy(i)-flwidth(i)) .le. lbdminwr ) then
            lbdminwr = (flmoy(i) - flwidth(i))/10.
          endif
          if ( (flmoy(i)+flwidth(i)) .ge. lbdmaxwr ) then
            lbdmaxwr = (flmoy(i) + flwidth(i))*100.
          endif
        enddo
ccccccccccccccccccccccccccc
c GALAXY  spectra
        nspmaxg=0
        nspmaxq=0
        nspmaxs=0
        nspmax =0
        nsp1=0
        nsp2=0
        nspfir=0
        nspphys=0
        if (chimin(1).lt.1e9 .and. zmin(1).ge.0) then
         call read_spec(gallib,reclmin(1),extilmin(1),extmin(1),zmin(1),
     >   extic,iext,opal,opat,iopa,
     >   lbdminwr,lbdmaxwr,3,wsp,fsp,nsp)
c
         if (addem(1:1).ne.'Y') then 
            do i = 1, nsp
              fgal(1,i)= fsp(i)-2.5*dlog10(dmmin(1)) + mod_distb(1)
              wgal(1,i)= wsp(i)
            enddo
            nsp1=nsp
         else
cccccccccccccccccccccccccccccccc
c  add Emission line in spectrum 
c    ... original SED without Emission lines 
             do i = 1, nsp
                fgal(2,i)= fsp(i)-2.5*dlog10(dmmin(1)) + mod_distb(1)
                wgal(2,i)= wsp(i)
             enddo 
             mabsuv= -2.5*physpara(2,reclmin(1)) + 51.61
             mabsuv=mabsuv-2.5*dlog10(dmmin(1))
c    ... building the Emission line spectrum 
             call addemlines(mabsuv,zmin(1),mod_distb(1),
     >                      aext_lb,extilmin(1),extmin(1),frac,
     >                       emspec,nem)
             do i = 1, nem
                 wq(i)= emspec(1,i)
                 fq(i)= emspec(2,i)
             enddo
c    ... resampling in the common wavelength range of the SED + Emission Lines spectra
             call sampling(wsp,fsp,nsp,wq,fq,nem,lsamp,y1,y2,lsmax)
c    ... rebuilding the entire SED spectrum + Emission lines
             lb = wsp(1)
             i=0
             do while ( lb .lt. lsamp(1) ) 
                i=i+1 
                wgal(1,i)= wsp(i)
                fgal(1,i)= fsp(i)-2.5*dlog10(dmmin(1))+mod_distb(1)
                lb=wgal(1,i)
             enddo 
             if (lb .gt. lsamp(1)) then 
                 i = i -1
             endif 
             do j = 1, lsmax
                 i = i + 1
                wgal(1,i)= lsamp(j)
                y1(j)=y1(j)-2.5*dlog10(dmmin(1))+mod_distb(1)
                if (y1(j).lt.100) then 
                   y1(j)=10**(-0.4*(y1(j)+48.59))   
                else
                   y1(j)=0.d0
                endif
                fgal(1,i) = y1(j)+y2(j)
                if (fgal(1,i).gt.1.d-60) then 
                   fgal(1,i)=-2.5*dlog10(y1(j)+y2(j))-48.59
                else
                   fgal(1,i)=90
                endif
             enddo  
             do j = 1, nsp
               if (wsp(j).gt.wgal(1,i)) then 
                 i=i+1 
                 wgal(1,i) = wsp(j)
                 fgal(1,i)= fsp(j)-2.5*dlog10(dmmin(1))+mod_distb(1)
               endif
             enddo
             nsp =  i 
             nsp1=nsp
         endif
c   End Add Emission Lines 
cccccccccccccccccccccccccccccccccccc        
         if (nsp.gt.nspmaxg )   nspmaxg=nsp
cccccccccccccccccccccccccccccccccccc        
c  2nd spectrum with 2nd solution exists 
         if (ndz.gt.1) then
            if (ndz.gt.2) ndz=2   ! just take the second spectra
            do k = 2, ndz  
              call read_spec(gallib,reclb(nb(k)),extilb(nb(k)),
     >          extb(nb(k)),zb(nb(k)),
     >          extic,iext,opal,opat,iopa,
     >          lbdminwr,lbdmaxwr,5,wsp,fsp,nsp) 
              ztemp=zb(nb(k))
              mod_dist2=funz(ztemp,h0,om0,l0)
              do i = 1, nsp
                fgal(2,i)= fsp(i)-2.5*dlog10(dmb(nb(k)))
     >                       +mod_dist2
                wgal(2,i)= wsp(i)
              enddo  
              if (nsp.gt.nspmaxg)    nspmaxg=nsp
            enddo
            nsp2=nsp
         endif
         if (nspmaxg.gt.nspmax) nspmax=nspmaxg
        endif
cccccccccccccccccccccccccccccccccccc        
c  FIR Library  Added to the 3rd  
        if (zmin(1).ge.0 .and. nlibext.ge.1 .and. reclfirb.gt.0) then
           call read_spec(libfir(libfirb),reclfirb,1,
     >           0.d0,zmin(1),
     >           extic,iext,opal,opat,iopa,
     >           lbdminwr,lbdmaxwr,2,wsp,fsp,nsp) 
c
           do i = 1, nsp
              fgal(3,i)= fsp(i)-2.5*dlog10(dmfirb)+mod_distb(1)
              wgal(3,i)= wsp(i)
           enddo  
          if (nsp.gt.nspmaxg) nspmaxg=nsp
          if (nspmaxg.gt.nspmax) nspmax=nspmaxg     
          nspfir=nsp
        endif
cccccccccccccccccccccccccccccccccccc        
c  PHYS library  added in 4th 
        if (zmin(1).ge.0 .and. nlibphys.eq.1 .and. reclpbest.gt.0) then
           call read_spec(libphys_sed,reclpbest,1,
     >           0.d0,zmin(1),
     >           extic,iext,opal,opat,iopa,
     >           lbdminwr,lbdmaxwr,5,wsp,fsp,nsp) 
c
           do i = 1, nsp
              fgal(4,i)= fsp(i)-2.5*dlog10(dmpbest)+mod_distb(1)
              wgal(4,i)= wsp(i)
           enddo  
          if (nsp.gt.nspmaxg) nspmaxg=nsp
          if (nspmaxg.gt.nspmax) nspmax=nspmaxg     
          nspphys=nsp
        endif
cccccccccccccccccccccccccccccccccccc        
c QSO
        if (chimin(2).lt.1e9 .and. zmin(2).ge.0) then
         call read_spec(qsolib,reclmin(2),1,0.d0,zmin(2),
     >    extic,iext,opal,opat,iopa,
     >    lbdminwr,lbdmaxwr,5,wsp,fsp,nsp)
c            ztemp=zmin(2)
c            mod_dist2=funz(ztemp,h0,om0,l0)
            do i = 1, nsp
              fq(i)= fsp(i)-2.5*dlog10(dmmin(2))+mod_distb(2)
              wq(i)= wsp(i)
            enddo
          if (nsp.ge.nspmaxq) nspmaxq=nsp
          if (nspmaxq.gt.nspmax) nspmax=nspmaxq
        endif
cccccccccccccccccccccccccccccccccccc        
c STAR
        if (chimin(3).lt.1e9 ) then
           call read_spec(starlib,reclmin(3),1,0.d0,0.d0,
     >     extic,iext,opal,opat,iopa,
     >     lbdminwr,lbdmaxwr,5,wsp,fsp,nsp)
           do i = 1, nsp
             fst(i)= fsp(i)-2.5*dlog10(dmmin(3)) 
             wst(i)= wsp(i)
           enddo
           if (nsp.ge.nspmaxs ) nspmaxs=nsp
           if (nspmaxs.gt.nspmax) nspmax=nspmaxs
        endif
cccccccccccccccccccccccccccccccccccc        
c WRITE OUTPUT SPECTRA
        open(7,file=ospec,status='unknown')
c   ... write Nfilt Zspec Zphot
c        write(7,'(i6,1x,2(f8.3,1x),E12.6)') imagm,zs,zmin(1),chimin(1)
        write(7,'(A)')  "# Ident Zspec,Zphot "
        write(7,'(i9,1x,2(f9.5,1x))') spec,zs,zmin(1)
c
        write(7,'(2A)') "# Mag emag  Lbd_mean  Lbd_width ",
     >                  "  Mag_gal  Mag_FIR  Mag_BCSTOCH "
        write(7,'(A,2x,i6)') "FILTERS  ",imagm

        write(7,'(A)') "# Zstep  PDF "
        if (fastmod(1:1).eq.'Y' .or. zfix(1:1).eq.'Y') then
           write(7,'(A,2x,I8)')  "PDF  ",0
        else 
           write(7,'(A,2x,I8)')  "PDF  ",chimax
        endif
c
        write(7,'(3(A,2x))') "# Type Nline Model Library Nband ",
     >                " Zphot Zinf Zsup Chi2  PDF  ",
     >                " Extlaw EB-V Lir Age  Mass SFR SSFR"
        if (zmin(1).ge.0.and.chimin(1).le.1.e9) then
          if (paramed(1).gt.0) paramed(1) = dlog10(paramed(1))
          write(7,706) "GAL-1 ",nsp1,imasmin(1),1,nbused,
     >    zmin(1),z68i,z68s,chimin(1),pdz(1),
     >    extilmin(1),extmin(1),paramed(2),
     >    paramed(1),paramed(4),paramed(5),paramed(6)  
        else
          write(7,706) "GAL-1 ",0,-1,-1,-1,
     >          -1.,-1.,-1.,-1.,-1.,
     >          -1,-1.,-1.,-1.,-1.,-1.,-1.
        endif  
c
        if (zmin(1).ge.0.and.chimin(1).le.1.e9 .and. ndz.gt.1) then
          write(7,706) "GAL-2 ",nsp2,imasb(nb(2)),1,nbused,
     >    zb(nb(2)),0.,0.,chi(2,nb(2)),pdz(2),
     >    extilb(nb(2)),extb(nb(2)),ldustb(nb(2)),
     >    ageb(nb(2)),-1.,-1.,-1. 
        else
          write(7,706) "GAL-2 ",0,-1,-1,-1,
     >          -1.,-1.,-1.,-1.,0.,
     >          -1,-1.,-1.,-1.,-1.,-1.,-1.
        endif  
        if (zmin(1).ge.0 .and. nlibext.ge.1 .and. reclfirb.gt.0) then
          write(7,706) "GAL-FIR ",nspfir,modfirb,libfirb,nf_fir,
     >         zmin(1),0.,0.,chi2_fir,0.,
     >         -1,-1.,lirmed,-1.,-1.,-1.,-1.
        else
          write(7,706) "GAL-FIR ",0,-1,-1,-1,
     >          -1.,-1.,-1.,-1.,-1.,
     >          -1,-1.,-1.,-1.,-1.,-1.,-1.
        endif  
c  ... write GAL-PHYS (BC07-STOCH)
        if (zmin(1).ge.0 .and. nlibphys.eq.1 .and. reclpbest.gt.0) then
          write(7,706) "GAL-STOCH ",nspphys,reclpbest-1,5,nbused,
     >      zmin(1),0.,0.,chipbest,0.,0,0.,ppmed(5),ppmed(1),
     >      ppmed(6),ppmed(8),ppmed(20)
        else
          write(7,706) "GAL-STOCH ",0,-1,-1,-1,
     >         -1.,-1.,-1.,-1.,-1.,
     >         -1,-1.,-1.,-1.,-1.,-1.,-1.
        endif  
c  ... write QSO 
        if (chimin(2).lt.1e9 .and. zmin(2).ge.0) then
          write(7,706) "QSO ",nspmaxq,imasmin(2),2,nbused,
     >      zmin(2),0.,0.,chimin(2),0.,-1,-1.,-1.,-1.,-1.,-1.,-1.
        else
          write(7,706) "QSO ",0,-1,-1,-1,
     >         -1.,-1.,-1.,-1.,-1.,
     >         -1,-1.,-1.,-1.,-1.,-1.,-1.
        endif  
c  ... write STAR
        if (chimin(3).lt.1e9) then
          write(7,706) "STAR ",nspmaxs,imasmin(3),3,nbused,
     >      0.,0.,0.,chimin(3),0.,-1,-1.,-1.,-1.,-1.,-1.,-1.
        else
          write(7,706) "STAR ",0,-1,-1,-1,
     >          -1.,-1.,-1.,-1.,-1.,
     >          -1,-1.,-1.,-1.,-1.,-1.,-1.
        endif  
c  ...  write mag obs + predicted + filters 
        do k = 1, imagm 
           if (magb(k).gt.0) then 
             magb(k) =  -2.5*dlog10(magb(k))-48.59
           else
             magb(k) = 99
           endif
           if (magtyp(1:1).eq.'V') then 
             ab(k)      = ab(k)      + abcor(k)
             magb(k)    = magb(k)    + abcor(k)
             fluxphys(k)= fluxphys(k)+ abcor(k)
           endif             
           write(7,701) ab(k),sab(k),flmoy(k),flwidth(k),
     >                  magb(k),magfirb(k),fluxphys(k)
        enddo
c  ... write PDF  
        if (fastmod(1:1).eq.'N' .and. zfix(1:1).eq.'N') then
           do i = 1,chimax
             val1 = chi(1,i)
             if (chimin(1).lt.1.e9)   val2 = yp(i)
             if (chimin(1).ge.1.e9)   val2 = chi(2,i)
             write(7,'(f9.5,2x,E12.6)') val1,val2
           enddo
        endif
c  ... write GAL 1st  
cc      Type Nline Model Lib Nbd Zphot Zinf Zsup Chi2 PDF Extlaw EB-V Lir  Age Mass SFR SSFR 
        if (zmin(1).ge.0.and.chimin(1).le.1.e9 .and. nsp1.gt.1) then
          do i = 1,nsp1
            write(7,707) wgal(1,i),fgal(1,i)
          enddo
        endif  
c  ... write GAL 2nd
        if (zmin(1).ge.0.and.chimin(1).le.1.e9 .and. ndz.gt.1
     >       .and. nsp2.gt.1) then
          do i = 1,nsp2
            write(7,707) wgal(2,i),fgal(2,i)
          enddo
        endif  
c  ... write GAL-FIR 
        if (zmin(1).ge.0 .and. nlibext.ge.1 .and. reclfirb.gt.0
     >       .and. nspfir.gt.0 ) then
          do i = 1,nspfir
            write(7,707) wgal(3,i),fgal(3,i)
          enddo
        endif  
c  ... write GAL-PHYS (BC07-STOCH)
        if (zmin(1).ge.0 .and. nlibphys.eq.1 .and. reclpbest.gt.0 
     >       .and.nspphys.gt.1 ) then
          do i = 1,nspphys
            write(7,707) wgal(4,i),fgal(4,i)
          enddo
        endif  
c  ... write QSO 
        if (chimin(2).lt.1e9 .and. zmin(2).ge.0 .and. nspmaxq.gt.1) then
          do i = 1,nspmaxq
            write(7,707) wq(i),fq(i)
          enddo
        endif  
c  ... write STAR
        if (chimin(3).lt.1e9 .and. nspmaxs.gt.1) then
          do i = 1,nspmaxs
            write(7,707) wst(i),fst(i)
          enddo
        endif  
c
        close(7)
c
      endif
c
cc        Type Nline Model Lib Nbd Zphot Zing Zsup Chi2 PDF Extlaw EB-V Lir Age Mass SFR SSFR  
 706   format(A,1x,4(I6,1x),3(F9.5,1x),E12.6,1x,f7.2,1x,
     >         I4,1x,2(f8.3,1x),E12.6,1x,3(f8.3,1x)) 
 707   format(1x,2(E14.8,2x))
ccccccccccccccccccccccccccccccc
c        if (ndz .eq. 1 .and. nlibext.eq.0) then
c          write(7,702) ndz,zmin(1),z68i,z68s,chimin(1),pdz(1),
c     >        dmmin(1),imasmin(1),agemin(1),extmin(1)
c        elseif (ndz.ge. 1 .and. nlibext.ge.1) then 
c          write(7,702) 2,zmin(1),z68i,z68s,chimin(1),pdz(1),
c     >        dmmin(1),imasmin(1),agemin(1),extmin(1)
cc            
c          write(7,705) libfirb,modfirb,zfirb,
c     >           lumfirb,lirmed,lirinf,lirsup,
c     >           chirmin,nf_fir,dmfirb  
cc
c        elseif (ndz.gt.1 .and. nlibext.eq.0 ) then                      
c          write(7,702) ndz,zmin(1),z68i,z68s,chimin(1),pdz(1),
c     >         dmmin(1),imasmin(1),agemin(1),extmin(1),
c     >         (zb(nb(k)),chi(2,nb(k)),pdz(k),dmb(nb(k)),
c     >         imasb(nb(k)),ageb(nb(k)),extb(nb(k)),k=2,ndz)
c        endif
c        write(7,703) zmin(2),chimin(2),dmmin(2),imasmin(2),
c     >                        imasmin(3),chimin(3)
c
c        if (nspmax.gt.wmax) nspmax=wmax   ! must be lower than size declaration
c        write(*,*) nspmax,chimax,wmax
c        do i = 1, max(nspmax,chimax)
c           if (i.gt.chimax) then
c              val1 = 0.
c              val2 = 0.
c           else
c              val1 = chi(1,i)
c              if (chimin(1).lt.1.e9)   val2 = yp(i)
c              if (chimin(1).ge.1.e9)   val2 = chi(2,i)
c           endif   
c           if (i.gt.nspmaxg) then
c              do k = 1, 5
c                 wgal(k,i) = 0.
c                 fgal(k,i) = 0.
c              enddo  
c           endif
c           if (i.gt.nspmaxq) then
c               wq(i) =0.
c               fq(i) =0.
c          endif
c           if (i.gt.nspmaxs) then
c               wst(i) =0.
c               fst(i) =0.
c           endif
c           if (nlibphys.eq.1)
c     >        write(7,704) val1,val2,wq(i),fq(i),wst(i),fst(i),
c     >                 (wgal(k,i),fgal(k,i),k=1,4)
c           if (nlibext.ge.1 .and. nlibphys.eq.0)
c     >        write(7,704) val1,val2,wq(i),fq(i),wst(i),fst(i),
c     >                 (wgal(k,i),fgal(k,i),k=1,3)
c           if (nlibext.eq.0 .and. nlibphys.eq.0)
c     >        write(7,704) val1,val2,wq(i),fq(i),wst(i),fst(i),
c     >                 (wgal(k,i),fgal(k,i),k=1,2)c
c
c        enddo
c        close(7)
c      endif
c      END of OUTPUT SPECTRA
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
 701  format(2(f10.3,1x),500(E14.6,2x))
c 702  format(i6,1x,3(f9.3,1x),3(E12.6,1x),1x,i8,1x,E12.6,1x,f9.3,
c     > 1x,70(f9.3,1x,3(E12.6,1x),i8,1x,E12.6,1x,f9.3,1x))
c 703  format(f9.3,1x,2(E12.6,1x),2(1x,i6),1x,E12.6)
c 704  format(f10.3,1x,E12.6,1x,16(E14.8,1x))
c 705  format(i6,1x,i8,2x,f5.3,2x,4(f8.3,1x),1x,E12.6,1x,i6,1x,f9.3)
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  LOOP ON THE NUMBER OF OBJECTS IN THE OBSERVED CATALOG 
 1    continue   ! iteration on the observ. catalogue 
      enddo
 8    nobjm = k - 1
      close(90)

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccccc   AUTO-ADAPT     ccccccccccccccccccc
c     If finished (convergence in auto-adapt is reached or no auto-adapt) 
      if(autoadapt(1:1).eq.'N' .or. realise.ge.2 )then
         write(UO,*)  '            '      
         write(UO,*)  ' Number of measured objects :'
         write(UO,'(4(A,I8,1x))') " GAL:",nmeas(1)," QSO:",nmeas(2),
     >"STAR:",nmeas(3)," over ",nobj 
         write(UO,*)  ' Results in file : ',outf(1:lnblnk(outf))
         write(UO,*)  ' That s all Folks !! '
      else
         ngals_ada=ngals_ada-1
c        keep free parameters before new step  
         do k=1,imagm
           a0in(k)=a0(k)
           a1in(k)=a1(k)
           a2in(k)=a2(k)
           a3in(k)=a3(k)
         enddo
c        main procedure for auto-adapt
         write(41,*)"ITERATION",iteration
         call auto_adapt(chiin,chifit,a0in,a0,a1in,a1,
     >                   a2in,a2,a3in,a3,realise,degre,adcont,
     >                   residu,adapterror)
c      new chi2
         chiin=chifit
c      write at each step the corrections applied and the min error
         do k = 1, imagm
            write(42,'(3(I4,1x),100(f12.5,1x))') realise,iteration,
     >      k,a0(k),a1(k),a2(k),a3(k),min_err(k)
         enddo

c      keep the value for the smallest chi2
         if(residu.le.res_best)then
            res_best=residu
            iter_best=iteration
            do k=1,imagm
               a0best(k)=a0(k)
               a1best(k)=a1(k)
               a2best(k)=a2(k)
               a3best(k)=a3(k)
               min_errbest(k)=min_err(k)
            enddo
         endif
c      stop iteration
         if(realise.ge.2)then
            write(UO,'(A)') "#####################################"
            write(UO,'(A)') " --> End of training : Corrections are:"
            write(UO,'(A20,1x,100(A9,1x))')
     >          "Filter","a0","a1","a2","a3"
            do k = 1, imagm
               write(UO,'(A20,1x,100(f9.3,1x))') 
     >          valf(k)(1:lnblnk(valf(k))),a0(k),a1(k),a2(k),a3(k) 
            enddo
            write(4,'(A)') "#####################################"
            write(4,'(A)') "# AUTO-ADAPT Coefficients:"
            write(4,'(A,1x,A20,1x,100(A9,1x))')
     >               "#"," Filter","a0","a1","a2","a3"
            do k = 1, imagm
               write(4,'(A,1x,A20,1x,100(f9.3,1x))') 
     >          "#",valf(k)(1:lnblnk(valf(k))),a0(k),a1(k),a2(k),a3(k) 
            enddo
            write(4,'(A,1x,100(f7.3,A))') 
     >            "# SHIFTS ",(a0(k),",",k=1,imagm)
            write(4,'(A)') "#####################################"
c
            write(UO,'(A)') "#####################################"
            write(UO,'(A)') " --> Running lephare ..."
         endif
         goto 6   !Redo the loop in auto-adapt
      endif
c      END   of autoadapt method 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      if(autoadapt(1:1).eq.'Y') then
         close(41)
         close(42)
      endif
c
c
      if (outpdz(1:4).ne."NONE") then 
        close(43)
        close(44)
        close(45)
        do i = 1,nfabs
          ufabs=45+i 
          close(ufabs)
        enddo
      endif

      close(4)
      if (UO.ne.6)   close(UO)
  
      STOP   
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  FORMATS 
c 530  format(1x,f6.3,1x,E12.6,1x,f9.6,1x,E12.6)
c 533  format(1x,I8,1x,i3,1x,4(f8.3,1x),2(1x,f11.6))
c 534  format(1x,I4,10(1x,E13.6))
c 535  format(3(1x,f6.3),1x,f10.3,1x,i8,1x,2(i3,1x),f9.5,1x,e12.6)
c 
c
 56   write (UO,*) 'File ',file(1:lnblnk(file)),' not found -> STOP '
      STOP
      end
