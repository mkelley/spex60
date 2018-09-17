import numpy as np
import scipy.ndimage as nd
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
from astropy.io import fits, ascii

__all__ = [
    'Spectrum',
    'PSGTrans'
]


class Spectrum:
    def __init__(self, wave, fluxd, unc, flags, meta):
        self.wave = wave
        self.fluxd = fluxd
        self.unc = unc
        self.flags = flags
        self.meta = meta

    @classmethod
    def from_file(cls, fn):
        spec, h = fits.getdata(fn, header=True)
        return Spectrum(spec[0], spec[1], spec[2], spec[3].astype(int), h)

    def __call__(self, wave):
        i = self.flags == 0
        fluxd = splev(wave, splrep(self.wave[i], self.fluxd[i]))
        unc = np.sqrt(splev(wave, splrep(self.wave, self.unc[i]**2)))
        return fluxd, unc

    def __mul__(self, scale):
        s = Spectrum(self.wave, self.fluxd * scale, self.unc * scale,
                     self.flags, self.meta)
        return s

    def __truediv__(self, denom):
        if isinstance(denom, Spectrum):
            i = self.flags == 0
            fluxd, unc = denom(self.wave[i])
            r = self.fluxd[i] / fluxd
            r_unc = np.sqrt(
                (unc / fluxd)**2 + (self.unc[i] / self.fluxd[i])**2)
            flags = np.zeros(sum(i), int)

            s = Spectrum(self.wave[i], r, r_unc, flags, self.meta)
            s.meta.add_history('Normalized by {OBJECT} at {TCS_AM}'.format(
                **denom.meta))
        else:
            s = self * (1 / denom)
        return s

    @property
    def normalized(self, wrange=[1.60, 1.70]):
        i = (self.wave > min(wrange)) * (self.wave < max(wrange))
        scale = np.mean(self.fluxd[i])
        return self / scale

    def plot(self, ax=None, **kwargs):
        ax = plt.gca() if ax is None else ax
        return ax.plot(self.wave, self.fluxd, **kwargs)

    def errorbar(self, ax=None, **kwargs):
        ax = plt.gca() if ax is None else ax
        return ax.errorbar(self.wave, self.fluxd, self.unc, **kwargs)

    def shift(self, dwave):
        i = self.flags == 0
        fluxd = splev(self.wave[i], splrep(
            self.wave[i] + dwave, self.fluxd[i]))
        var = splev(self.wave[i], splrep(self.wave[i] + dwave, self.unc[i]**2))
        unc = np.sqrt(var)
        self.meta['wshift'] = self.meta.get('wshift', 0) + dwave
        return Spectrum(self.wave, fluxd, unc, self.flags, self.meta)

    def smooth(self, width):
        i = self.flags == 0
        fluxd = nd.gaussian_filter(self.fluxd, width)
        return Spectrum(self.wave, fluxd, self.unc, self.flags, self.meta)


class PSGTrans(Spectrum):
    def __init__(self, wave, trans, components={}):
        self.wave = wave
        self.trans = trans
        self.components = components

    @property
    def fluxd(self):
        return self.trans

    @property
    def unc(self):
        return np.zeros_like(self.wave)

    @property
    def flags(self):
        return np.zeros_like(self.wave, int)

    def __call__(self, wave):
        return super().__call__(wave)[0]

    def __truediv__(self, denom):
        if isinstance(denom, PSGTrans):
            i = self.flags == 0
            d = denom(self.wave[i])
            r = self.fluxd[i] / d

            s = PSGTrans(self.wave[i], r)
        else:
            s = self * (1 / denom)
        return s

    @classmethod
    def from_file(cls, fn):
        names = ('wave', 'total', 'H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4',
                 'O2', 'N2', 'Rayleigh', 'CIA')
        spec = ascii.read(fn, names=names)

        components = {}
        for col in names[2:]:
            components[col] = spec[col].data

        return PSGTrans(spec['wave'].data, spec['total'].data, components)

    def smooth(self, width):
        i = self.flags == 0
        trans = nd.gaussian_filter(self.trans, width)
        return PSGTrans(self.wave, trans)
