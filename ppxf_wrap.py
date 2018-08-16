from ppxf import ppxf
import ppxf_util as util
class ppxf_wrap():
    def __init__(self, redshift, wave, specres):
        wave=wave/(1+redshift) #  When measure velocity from IFS observation, 
                                        # deredshift to the barycenter should be applied before measuement.
        specres=specres/(1+redshift)
        
# Only use the wavelength range in common between galaxy and stellar library.
#         mask = (t['loglam'] > np.log10(3540)) & (t['loglam'] < np.log10(7409))     
        mask = (wave > 3540) & (wave < 7409)
        loglam_gal = np.log10(wave[mask])
        lam_gal = wave[mask]
        specres=specres[mask]

        c = 299792.458                  # speed of light in km/s
        frac = lam_gal[1]/lam_gal[0]    # Constant lambda fraction per pixel
        fwhm_gal=lam_gal/specres          # Resolution FWHM of every pixel, in Angstroms
        velscale = np.log(frac)*c       # Constant velocity scale in km/s per pixel

        file_dir='./'
        galaxy_templates=glob.glob(file_dir+'Mun1.30Z*.fits')
        fwhm_tem=2.51 # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
        hdu = fits.open(galaxy_templates[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
        lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
        sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
        templates = np.empty((sspNew.size, len(galaxy_templates)))
        dv = np.log(lam_temp[0]/lam_gal[0])*c    # km/s
        
        fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gal)        
        fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
        sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels
        
        for j, fname in enumerate(galaxy_templates):
            hdu = fits.open(fname)
            ssp = hdu[0].data
            ssp = util.gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma
            sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
            templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates
                
        self.templates=templates
        self.flux=None
        self.ivar=None
        self.specres=specres
        self.mask=mask
        self.lam_gal=lam_gal
        self.dv=dv
        self.lamRange_temp=lamRange_temp
        self.velscale=velscale
        
    def run(self):
        flux=(self.flux)[self.mask]
        noise=(self.ivar**(-0.5))[self.mask]
        
        specres=self.specres
        templates=self.templates
        lam_gal=self.lam_gal
        nmask=(np.isfinite(noise) & (noise > 0))
        dv=self.dv
        velscale=self.velscale

            
        flux = flux[nmask]
        galaxy = flux/np.median(flux)   # Normalize spectrum to avoid numerical issues
        noise = noise[nmask]/np.median(flux)
        lam_gal=lam_gal[nmask]
        goodpixels = util.determine_goodpixels(np.log(lam_gal), self.lamRange_temp, 0)

# Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
#         vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
        start = [0, 200.]  # (km/s), starting guess for [V, sigma]
        
        adegree=0
        mdegree=0
        pp=ppxf(templates, galaxy, noise, velscale, start,
                  goodpixels=goodpixels, plot=False, quiet=True, moments=2,
                  degree=adegree, mdegree=mdegree, vsyst=dv, clean=False, lam=lam_gal)
        self.fflux=galaxy
        return pp
