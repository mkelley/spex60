# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import re

import numpy as np

import astropy.units as u
from astropy.io import ascii, fits

from .config import config

__all__ = ['SpeX', 'Prism60']


class Calibration:
    def __init__(self):
        calpath = os.path.join(
            config['spextool_path'], 'instruments', 'uspex', 'data')
        self.bpm = fits.getdata(os.path.join(calpath, 'uSpeX_bdpxmk.fits'))

        self.lincoeff = fits.getdata(
            os.path.join(calpath, 'uSpeX_lincorr.fits'))

        fn = os.path.join(calpath, 'uSpeX_bias.fits')
        self.bias = fits.getdata(fn) / SpeX.read_header(fn)['DIVISOR']

        self.linecal = fits.getdata(
            os.path.join(calpath, 'Prism_LineCal.fits'))
        # Prism_LineCal header is bad and will throw a warning
        self.linecal_header = SpeX.read_header(
            os.path.join(calpath, 'Prism_LineCal.fits'))

        self.lines = ascii.read(os.path.join(calpath, 'lines.dat'),
                                names=('wave', 'type'))


class SpeX:
    """Basic SpeX file IO."""

    def __init__(self, *args, **kwargs):
        self.cal = Calibration()

        self.mask = ~self.cal.bpm.astype(bool)
        self.mask[:, :config['x range'][0]] = 1
        self.mask[:, config['x range'][1]:] = 1
        self.mask[:config['bottom']] = 1
        self.mask[config['top']:] = 1

        self.flat = None
        self.flat_var = None
        self.flat_h = None

    def read(self, files, pair=False, ampcor=True, lincor=True, flatcor=True,
             abba_test=True):
        """Read uSpeX files.

        Parameters
        ----------
        files : string or list
          A file name or list thereof.
        pair : bool, optional
          Assume the observations are taken in AB(BA) mode and return
          A-B for each pair.
        ampcor : bool optional
          Set to `True` to apply the amplifcation noise correction.
        lincor : bool, optional
          Set to `True` to apply the linearity correction.
        flatcor : bool, optional
          Set to `True` to apply flat field correction.
        abba_test : bool, optional
          Set to `True` to test for AB(BA) ordering when `pair` is
          `True`.  If `abba_test` is `False`, then the file order is
          not checked.

        Returns
        -------
        stack : MaskedArray
          The resultant image(s).  [counts / s]
        var : MaskedArray
          The variance.  [total DN]
        headers : list or astropy FITS header
          If `pair` is `True`, the headers will be a list of lists,
          where each element is a list containing the A and B headers.

        """

        from numpy.ma import MaskedArray

        if isinstance(files, (list, tuple)):
            print('Loading {} files.'.format(len(files)))
            stack = MaskedArray(np.empty((len(files), 2048, 2048)))
            var = MaskedArray(np.empty((len(files), 2048, 2048)))
            headers = []
            for i in range(len(files)):
                kwargs = dict(pair=False, ampcor=ampcor, lincor=lincor,
                              flatcor=flatcor)
                stack[i], var[i], h = self.read(files[i], **kwargs)
                headers.append(h)

            if pair:
                print('\nAB(BA) pairing and subtracting.')
                a = np.flatnonzero(
                    np.array([h['BEAM'] == 'A' for h in headers]))
                b = np.flatnonzero(
                    np.array([h['BEAM'] == 'B' for h in headers]))
                if abba_test:
                    # require equal numbers of a and b
                    if len(a) != len(b):
                        raise ValueError('Number of A beams not equal to'
                                         ' number of B beams')
                    # each A-B pair should be number neighbors
                    for i, j in zip(a, b):
                        if abs(i - j) != 1:
                            raise ValueError('Found invalid A-B pair: '
                                             + headers[i]['IRAFNAME'] + ' '
                                             + headers[j]['IRAFNAME'])

                stack_AB = []
                var_AB = []
                headers_AB = []
                for i, j in zip(a, b):
                    stack_AB.append(stack[i] - stack[j])
                    var_AB.append(var[i] + var[j])
                    headers_AB.append([headers[i], headers[j]])

                stack_AB = np.ma.MaskedArray(stack_AB)
                var_AB = np.ma.MaskedArray(var_AB)
                return stack_AB, var_AB, headers_AB
                # if abba_test:
                #    # Require ABBA ordering
                #    if not all([h['BEAM'] == 'A' for h in headers[::4]]):
                #        raise ValueError('Files not in an ABBA sequence')
                #    if not all([h['BEAM'] == 'B' for h in headers[1::4]]):
                #        raise ValueError('Files not in an ABBA sequence')
                #    if not all([h['BEAM'] == 'B' for h in headers[2::4]]):
                #        raise ValueError('Files not in an ABBA sequence')
                #    if not all([h['BEAM'] == 'A' for h in headers[3::4]]):
                #        raise ValueError('Files not in an ABBA sequence')

                # fancy slicing, stacking, and reshaping to get:
                #   [0 - 1, 3 - 2, 4 - 5, 7 - 6, ...]
                # stack_A = stack[::4] - stack[1::4]  # A - B
                # var_A = var[::4] + var[1::4]
                # headers_A = [[a, b]
                #             for a, b in zip(headers[::4], headers[1::4])]
                # if len(files) > 2:
                #    stack_B = stack[3::4] - stack[2::4]  # -(B - A)
                #    stack = np.ma.vstack((stack_A, stack_B))

                #    var_B = var[2::4] + var[3::4]
                #    var = np.ma.vstack((var_A, var_B))

                #    headers_B = [[a, b]
                #                 for a, b in zip(headers[3::4], headers[2::4])]
                #    headers = [None] * (len(headers_A) + len(headers_B))
                #    headers[::2] = headers_A
                #    headers[1::2] = headers_B
                # else:
                #    stack = stack_A[0]
                #    var = var_A[0]
                #    headers = headers_A[0]

            # return stack, var, headers

        print('Reading {}'.format(files))
        data = fits.open(files, lazy_load_hdus=False)
        data[0].verify('silentfix')

        # check if already processed
        if 'SPEX60' in data[0].header:
            mask = data['mask'].astype(bool)
            im = np.ma.MaskedArray(data['sci'].data, mask=mask)
            var = data['var'].data
            if 'b header' in data:
                h = [data['sci'].header, data['b header'].header]
            else:
                h = data['sci'].header
            data.close()
            return im, var, h

        h = data[0].header.copy()
        read_var = (2 * config['readnoise']**2
                    / h['NDR']
                    / h['CO_ADDS']
                    / h['ITIME']**2
                    / config['gain']**2)

        # TABLE_SE is read time, not sure what crtn is.
        crtn = (1 - h['TABLE_SE'] * (h['NDR'] - 1)
                / 3.0 / h['ITIME'] / h['NDR'])
        t_exp = h['ITIME'] * h['CO_ADDS']

        im_p = data[1].data / h['DIVISOR']
        im_s = data[2].data / h['DIVISOR']
        data.close()

        mask_p = im_p < (self.cal.bias - config['lincor max'])
        mask_s = im_s < (self.cal.bias - config['lincor max'])
        mask = mask_p + mask_s
        h.add_history('Masked saturated pixels.')

        im = MaskedArray(im_p - im_s, mask)

        if ampcor:
            im = self._ampcor(im)
            h.add_history('Corrected for amplifier noise.')

        if lincor:
            cor = self._lincor(im)
            cor[mask] = 1.0
            cor[:4] = 1.0
            cor[:, :4] = 1.0
            cor[2044:] = 1.0
            cor[:, 2044:] = 1.0
            im /= cor
            h.add_history('Applied linearity correction.')

        if flatcor:
            if self.flat is None:
                raise ValueError(
                    "Flat correction requested but flat not loaded.")
            im /= self.flat
            h.add_history('Flat corrected.')

        # total DN
        var = (np.abs(im * h['DIVISOR'])
               * crtn
               / h['CO_ADDS']**2
               / h['ITIME']**2
               / config['gain']
               + read_var)  # / h['DIVISOR']**2 / h['ITIME']**2
        # counts / s
        im = im / h['ITIME']
        im.mask += self.mask
        return im, var, h

    def read_numbered(self, files, numbered=None, between=None, **kwargs):
        """Read from list based on observation number.

        Requires a file name format that ends with "N.a.fits" or
        "N.b.fits" where N is an integer.

        Parameters
        ----------
        files : list
            List of file names to consider.

        numbered : list
            List of observation numbers to read.

        between : list
            Read files with observation number starting with
            ``min(between)`` and ending with ``max(between)``
            (inclusive).  May be a list of lists for multiple sets.

        **kwargs
            Keyword arguments to pass to ``SpeX.read``.

        """

        def number(files):
            pat = re.compile('([0-9]+)\.[ab]\.fits$')
            for f in files:
                m = re.findall(pat, f)
                if m:
                    yield int(m[0]), f

        def find_between(files, between):
            read_list = []
            between = min(between), max(between)
            for n, f in number(files):
                if n >= between[0] and n <= between[1]:
                    read_list.append(f)
            return read_list

        read_list = []
        if numbered is not None:
            for n, f in number(files):
                if n in numbered:
                    read_list.append(f)
        elif between is not None:
            if isinstance(between[0], (list, tuple)):
                for b in between:
                    read_list += find_between(files, b)
            else:
                read_list += find_between(files, between)
        else:
            raise ValueError(
                'One of ``numbered`` or ``between`` must be provided.')

        return self.read(read_list, **kwargs)

    @classmethod
    def read_header(cls, filename, ext=0):
        """Read a header from a SpeX FITS file.

        SpeX headers tend to be missing quotes around strings.  The
        header will be silently fixed.

        """
        inf = fits.open(filename)
        inf[0].verify('silentfix')
        h = inf[0].header.copy()
        inf.close()
        return h

    def median_combine(self, stack, variances, headers, scale=False,
                       **kwargs):
        """Median combine a set of SpeX images or spectra.

        Parameters
        ----------
        stack : MaskedArray
        variances : MaskedArray
        headers : list
            From ``SpeX.read()``.

        **kwargs
            ``sigma_clip`` keyword arguments.

        Returns
        -------
        data : MaskedArray
        var : MaskedArray
            Combined data and variance.
        header : list or astropy.fits.Header
            Annotated header(s), based on the first item in the stack.

        Notes
        -----
        ITIME and HISTORY are updated in the header.

        """

        from astropy.stats import sigma_clip

        if isinstance(stack, (list, tuple)):
            clip = sigma_clip(stack, **kwargs)
        else:
            clip = sigma_clip(stack, axis=0, **kwargs)

        data = np.ma.median(clip, 0)
        var = np.ma.mean(variances, 0)
        header = headers[0]
        if isinstance(header, list):
            for i in range(2):
                header[i]['ITIME'] = sum([h[i]['ITIME'] for h in headers])
                files = [h[i]['IRAFNAME'] for h in headers]
                header[i].add_history(
                    'Median combined files: ' + ','.join(files))
        else:
            header['ITIME'] = sum([h['ITIME'] for h in headers])
            files = [h['IRAFNAME'] for h in headers]
            header.add_history('Median combined files: ' + ','.join(files))

        return data, var, header

    def save_image(self, im, var, h, filename=None, path=''):
        """Save SpeX image data."""

        from . import __version__

        if len(path) > 0 and not os.path.exists(path):
            os.mkdir(path)

        if filename is None:
            filename = os.path.join(path, h['IRAFNAME'])

        if isinstance(h, list):
            header = h[0]
        else:
            header = h

        hdu = fits.HDUList()
        hdu.append(fits.PrimaryHDU(header=header))
        hdu[0].header['SPEX60'] = __version__
        hdu.append(fits.ImageHDU(im.data, header=header, name='sci'))
        hdu.append(fits.ImageHDU(im.mask.astype(int), name='mask'))
        hdu.append(fits.ImageHDU(var.data, name='var'))

        if isinstance(h, list):
            hdu.append(fits.ImageHDU(header=h[1], name='b header'))
            hdu[3].header['SPEX60'] = __version__

        hdu.writeto(filename, output_verify='silentfix', overwrite=True)


class Prism60(SpeX):
    """Reduce uSpeX 60" prism data."""

    def __init__(self, *args, **kwargs):
        self.wavecal = None
        self.wavecal_h = None

        self.wave = None
        self.spec = None
        self.var = None
        self.rap = None
        self.bgap = None
        self.bgorder = None

        SpeX.__init__(self, *args, **kwargs)

    def _edges(self, im, order=2, plot=False):
        """Find the edges of the spectrum using a flat.

        Parameters
        ----------
        im : ndarray
          A flat field.
        order : int, optional
          The order of the polynomial fit to determine the edge.
        plot : bool, optional
          If `True`, show the image and edges in a matplotlib window.

        Returns
        -------
        b, t : ndarray
          The polynomial coefficients of the bottom and top edges,
          e.g., `bedge = np.polyval(b, x)`

        Notes
        -----
        Based on mc_findorders in Spextool v4.1 (M. Cushing).

        """

        from mskpy import image
        import scipy.ndimage as nd

        y = image.yarray(im.shape)
        x = image.xarray(im.shape)

        binf = 4

        def rebin(im): return np.mean(
            im.reshape((im.shape[0], binf, im.shape[1] / binf)),
            1).astype(int)

        rim = rebin(im)
        ry = rebin(y)
        rx = rebin(x)

        # find where signal falls to 0.75 x center
        i = int(np.mean(config['y range']))
        fcen = rim[i]

        bguess, tguess = np.zeros((2, rim.shape[1]), int)
        for j in range(rim.shape[1]):
            bguess[j] = np.min(ry[:i, j][rim[:i, j] > 0.75 * fcen[j]])
            tguess[j] = np.max(ry[i:, j][rim[i:, j] > 0.75 * fcen[j]])

        # scale back up to 2048
        bguess = np.repeat(bguess, binf)
        tguess = np.repeat(tguess, binf)

        # find actual edge by centroiding (center of mass) on Sobel
        # filtered image
        def center(yg, y, sim):
            s = slice(yg - 5, yg + 6)
            yy = y[s]
            f = sim[s]
            return nd.center_of_mass(f) + yy[0]

        sim = np.abs(nd.sobel(im * 1000 / im.max(), 0))
        bcen, tcen = np.zeros((2, im.shape[1]))
        for i in range(*config['x range']):
            bcen[i] = center(bguess[i], y[:, i], sim[:, i])
            tcen[i] = center(tguess[i], y[:, i], sim[:, i])

        def fit(x, centers):
            A = np.vstack((x**2, x, np.ones(len(x)))).T
            return np.linalg.lstsq(A, centers)[0]

        xx = np.arange(im.shape[1])
        s = slice(*config['x range'])
        b = fit(xx[s], bcen[s])
        t = fit(xx[s], tcen[s])

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 8))
            ax = plt.gca()
            ax.imshow(im, cmap=plt.cm.gray)
            ax.plot(xx[::binf], bcen[::binf], 'gx')
            ax.plot(xx[::binf], tcen[::binf], 'gx')
            ax.plot(xx, np.polyval(b, xx), 'r-')
            ax.plot(xx, np.polyval(t, xx), 'r-')
            plt.draw()

        return b, t

    def _ampcor(self, im):
        """Correct an image for amplifier noise.

        The median of the reference pixels is subtracted from each
        amplifier column.

        Notes
        -----
        Based on mc_findorders in Spextool v4.1 (M. Cushing).

        """

        amps = np.rollaxis(im[2044:].reshape((4, 64, 32)), 1).reshape(64, 128)
        m = np.median(amps, 1)
        return im - np.tile(np.repeat(m, 32), 2048).reshape(im.shape)

    def _lincor(self, im):
        """Linearity correction for an image.

        Notes
        -----
        Following mc_imgpoly from Spextool v4.1 (M. Cushing).

        """
        y = 0
        for i in range(len(self.cal.lincoeff)):
            y = y * im + self.cal.lincoeff[-(i + 1)]
        return y

    def process_cal(self, files, path=None, overwrite=False):
        """Generate flat field and wavelength calibration files.

        Parameters
        ----------
        files : list
          A list of images created by SpeX calibration macros.  Only
          60" prism data will be considered.
        path : string, optional
          Save files to this directory.  If `None`, they will be saved
          to "cal-YYYYMMDD".
        overwrite : bool, optional
          Set to `True` to overwrite previous calibration files.

        """

        # flats
        flats = sorted([f for f in files
                        if os.path.basename(f).startswith('flat')])
        fset = []
        n_sets = 0
        first_n = -1
        last_n = -1
        for i in range(len(flats)):
            h = self.read_header(flats[i])
            tests = (h['OBJECT'] == 'Inc lamp',
                     h['GRAT'] == 'Prism',
                     'x60' in h['SLIT'])
            if not all(tests):
                fset = []
                continue

            m = re.findall('flat-([0-9]+).a.fits$', flats[i])
            if len(m) != 1:
                raise ValueError('Cannot parse file name: {}'.format(flats[i]))
            n = int(m[0])

            fset.append(flats[i])
            print('Found {}  ({})'.format(flats[i], len(fset)))

            if len(fset) == 1:
                first_n = n
            elif len(fset) > 2:
                step = n - last_n
                last_n = n
                if step != 1:
                    print('  Bad image sequence.')
                    fset = []
                    continue
            else:
                last_n = n

            if len(fset) == 5:
                if path is None:
                    _path = 'cal-' + h['DATE_OBS'].replace('-', '')
                else:
                    _path = path

                try:
                    os.mkdir(_path)
                except FileExistsError:
                    pass

                fn = '{}/flat-{:05d}-{:05d}.fits'.format(
                    _path, first_n, last_n)
                if os.path.exists(fn):
                    print(fn, 'already exists, skipping')
                    fset = []
                    continue

                self.load_flat(fset)
                outf = fits.HDUList()
                outf.append(fits.PrimaryHDU(self.flat, self.flat_h))
                outf.append(fits.ImageHDU(self.flat_var, name='var'))
                outf.writeto(fn, output_verify='silentfix', clobber=overwrite)

                fset = []

        # done with flats

        # arcs
        arcs = sorted([f for f in files
                       if os.path.basename(f).startswith('arc')])
        for i in range(len(arcs)):
            h = self.read_header(arcs[i])
            tests = (h['OBJECT'] == 'Argon lamp',
                     h['GRAT'] == 'Prism',
                     'x60' in h['SLIT'])
            if not all(tests):
                continue

            m = re.findall('arc-([0-9]+).a.fits$', arcs[i])
            if len(m) != 1:
                raise ValueError('Cannot parse file name: {}'.format(arcs[i]))
            n = int(m[0])

            # only one arc lamp observation per Prism 60" cal
            if path is None:
                _path = 'cal-' + h['DATE_OBS'].replace('-', '')
            else:
                _path = path

            try:
                os.mkdir(_path)
            except FileExistsError:
                pass

            fn = '{}/wavecal-{:05d}.fits'.format(_path, n)
            self.load_wavecal(arcs[i])
            fits.writeto(fn, self.wavecal, self.wavecal_h,
                         output_verify='silentfix', clobber=overwrite)
        # done with arcs

    def load_flat(self, files):
        """Generate or read in a flat.

        Parameters
        ----------
        files : list or string
          A list of file names of data taken with the SpeX cal macro,
          or the name of an already prepared flat.
        save : bool, optional
          If `True`, save the new flat and variance data as a FITS
          file.  The name will be generated from the file list
          assuming they originated from a SpeX calibration macro.

        """

        from numpy.ma import MaskedArray
        import scipy.ndimage as nd
        from scipy.interpolate import splrep, splev

        if isinstance(files, str):
            self.flat, self.flat_h = fits.getdata(files, header=True)
            self.flat_var = fits.getdata(files, ext=1)
            return

        stack, headers = self.read(files, flatcor=False)[::2]
        h = headers[0]
        scale = np.array([np.ma.median(im) for im in stack])
        scale /= np.mean(scale)
        for i in range(len(stack)):
            stack[i] /= scale[i]

        flat = np.median(stack, 0)
        var = np.var(stack, 0) / len(stack)

        c = np.zeros(flat.shape)
        x = np.arange(flat.shape[1])
        for i in range(flat.shape[1]):
            if np.all(flat[i].mask):
                continue
            j = ~flat[i].mask
            y = nd.median_filter(flat[i][j], 7)
            c[i] = splev(x, splrep(x[j], y))

        h.add_history('Flat generated from: ' + ' '.join(files))
        h.add_history(
            'Images were scaled to the median flux value, then median combined.  The variance was computed then normalized by the number of images.')

        self.flat = (flat / c).data
        self.flat_var = (var / c).data
        self.flat_h = h

    def load_wavecal(self, fn, plot=False, debug=False):
        """Load or generate a wavelength calibration.

        Parameters
        ----------
        fn : string
          The name of an arc file taken with the SpeX cal macro or an
          already prepared wavelength calibration.
        plot : bool, optional
          Set to `True` to plot representative wavelength solutions.

        Notes
        -----
        Based on Spextool v4.1 (M. Cushing).

        """

        import scipy.ndimage as nd
        from mskpy import util, image

        h = self.read_header(fn)
        if 'wavecal' in h:
            if h['wavecal'] != 'T':
                raise ValueError(
                    "WAVECAL keyword present in FITS header, but is not 'T'.")
            print('Loading stored wavelength solution.')
            wavecal = fits.getdata(fn)
            mask = ~np.isfinite(wavecal)
            self.wavecal = np.ma.MaskedArray(wavecal, mask=mask)
            self.wavecal_h = h
            return

        arc = self.read(fn, flatcor=False)[0]

        slit = h['SLIT']
        slitw = float(slit[slit.find('x') + 1:])

        flux_anchor = self.cal.linecal[1]
        wave_anchor = self.cal.linecal[0]
        offset = np.arange(len(wave_anchor)) - int(len(wave_anchor) / 2.0)
        self.wavecal = image.xarray(arc.shape, dtype=float)

        disp_deg = self.cal.linecal_header['DISPDEG']
        w2p = []
        p2w = []
        for i in range(disp_deg + 1):
            w2p.append(self.cal.linecal_header['W2P01_A{}'.format(i)])
            p2w.append(self.cal.linecal_header['P2W_A0{}'.format(i)])

        xr = slice(*config['x range'])
        # xr = slice(wave_anchor[0], wave_anchor[1])
        xcor_offset = np.zeros(2048)
        for i in range(config['bottom'], config['top']):
            spec = arc[i, xr]

            xcor = nd.correlate(spec, flux_anchor, mode='constant')
            j = np.argmax(xcor)
            s = slice(j - 7, j + 8)
            guess = (np.max(xcor), offset[j], 5, 0.0)

            xx = offset[s]
            yy = xcor[s]
            j = np.isfinite(xx * yy)
            fit = util.gaussfit(xx[j], yy[j], None, guess)
            xcor_offset[i] = fit[0][1]

        # smooth out bad fits
        i = xcor_offset != 0
        y = np.arange(2048)
        p = np.polyfit(y[i], xcor_offset[i], 2)
        r = np.abs(xcor_offset - np.polyval(p, y))

        i = (xcor_offset != 0) * (r < 1)
        p = np.polyfit(y[i], xcor_offset[i], 2)

        # update wave cal with solution
        for i in range(config['bottom'], config['top']):
            x = self.wavecal[i, xr] - np.polyval(p, y[i])
            self.wavecal[i, xr] = np.polyval(p2w[::-1], x)

        h['wavecal'] = 'T'
        h['bunit'] = 'um'

        self.wavecal[self.mask] = np.nan
        self.wavecal_h = h
        self.arc = arc

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.clf()
            plt.plot(np.polyval(p2w[::-1], wave_anchor), flux_anchor,
                     'k-', lw=1.5)
            lax = plt.gca().twinx()
            for y in (700, 920, 1150):
                lax.plot(self.wavecal[y, xr], arc[y, xr])
            plt.draw()

        if debug:
            return offset, xcor_offset

    def _profile(self, im):
        """Create the image spatial profile.

        Currently just an unintelligent median.

        Parameters
        ----------
        im : ndarray or MaskedArray
          The 2D spectral image or an array of 2D images.

        Returns
        -------
        profile : MaskedArray
          The spatial profile.

        """

        return np.ma.median(im, 1)

    def peak(self, im, mode='AB', rap=5, smooth=0, plot=True,
             ex_rap=None, bgap=None):
        """Find approximate locations of profile peaks in a spatial profile.

        The strongest peaks are found via centroid on the profile
        min/max.

        Parameters
        ----------
        im : ndarray or MaskedArray
          The 2D spectral image.
        mode : string, optional
          'AB' if there is both a positive and a negative peak.  Else,
          set to 'A' for a single positive peak.
        rap : int, optional
          Radius of the fitting aperture.
        smooth : float, optional
          Smooth the profile with a `smooth`-width Gaussian before
          searching for the peak.
        plot : bool, optional
          Plot results.
        ex_rap : float
          Show this extraction aperture radius in the plot.
        bgap : array of two floats
          Show this extraction background aperture in the plot.

        Result
        ------
        self.peaks : ndarray
          The peaks.  For a stack: NxM array where N is the number of
          images, and M is the number of peaks.

        """

        import scipy.ndimage as nd
        from mskpy.util import between, gaussfit

        profile = self._profile(im)
        if smooth > 0:
            profile = nd.gaussian_filter(profile, smooth)

        self.peaks = []
        x = np.arange(len(profile))

        i = between(x, profile.argmax() + np.r_[-rap, rap])
        c = nd.center_of_mass(profile[i]) + x[i][0]
        self.peaks.append(c[0])

        if mode.upper() == 'AB':
            i = between(x, profile.argmin() + np.r_[-rap, rap])
            c = nd.center_of_mass(-profile[i]) + x[i][0]
            self.peaks.append(c[0])

        self.peaks = np.array(self.peaks)

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.clear()
            ax = fig.add_subplot(111)
            ax.plot(profile, color='k')
            for p in self.peaks:
                ax.axvline(p, color='r')

                if ex_rap is not None:
                    i = between(x, p + np.r_[-1, 1] * ex_rap)
                    ax.plot(x[i], profile[i], color='b', lw=3)

                if bgap is not None:
                    for s in [-1, 1]:
                        i = between(x, np.sort(p + s * np.r_[bgap]))
                        ax.plot(x[i], profile[i], color='c', lw=3)

            fig.canvas.draw()
            fig.show()

    def trace(self, im, plot=True):
        """Trace the peak(s) of an object.

        Best executed with standard stars.

        Initial peak guesses taken from `self.peaks`.  If there are
        multiple, even-indexed peaks are assumed to be the positive,
        odd-indexed peaks are assumed to be the negative beam.

        Parameters
        ----------
        im : MaskedArray
          The 2D spectrum to trace.
        plot : bool, optional
          Plot results.

        Result
        ------
        self.traces : list of ndarray
          The traces of each peak.
        self.trace_fits : list of ndarray
          The best-fit polynomical coefficients of the traces.

        """

        from mskpy import image

        profile = self._profile(im)
        self.traces = []
        self.trace_fits = []
        for i in range(len(self.peaks)):
            s = (-1)**i
            guess = ((s * profile).max(), self.peaks[i], 2.)
            trace, fit = image.trace(s * im, None, guess, rap=10,
                                     polyfit=True, order=7)
            fit = np.r_[fit[:-1], fit[-1] - self.peaks[i]]
            self.traces.append(trace)
            self.trace_fits.append(fit)

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.clear()
            ax = fig.add_subplot(111)

            for i in range(len(self.traces)):
                j = ~self.traces[i].mask
                x = np.arange(len(self.traces[i]))
                ax.plot(x, self.traces[i], color='k', marker='+', ls='none')
                fit = np.r_[self.trace_fits[i][:-1],
                            self.trace_fits[i][-1] + self.peaks[i]]
                ax.plot(x[j], np.polyval(fit, x[j]), color='r')

            fig.canvas.draw()
            fig.show()

    def _aper(self, y, trace, rap, subsample):
        """Create an aperture array for `extract`."""
        aper = (y >= trace - rap) * (y <= trace + rap)
        aper = aper.reshape(y.shape[0] // subsample, subsample, y.shape[1])
        aper = aper.sum(1) / subsample
        return aper

    def extract(self, im, h, rap, bgap=None, bgorder=0, var=None, traces=True,
                abcombine=True, append=False):
        """Extract a spectrum from an image.

        Extraction positions are from `self.peaks`.

        See `image.spextract` for implementation details.

        Parameters
        ----------
        im : MaskedArray
          The 2D spectral image, or array thereof.
        h : astropy FITS header
          The header for im.
        rap : float
          Aperture radius.
        bgap : array, optional
          Inner and outer radii for the background aperture, or `None`
          for no background subtraction.
        bgorder : int, optional
          Fit the background with a `bgorder` polynomial.
        var : MaskedArray, optional
          The variance image.  Used when `bgap` is not provided.
        traces : bool, optional
          Use `self.traces` for each peak.
        abcombine : bool, optional
          Combine (sum) apertures as if they were AB pairs.  The
          B-beam will be linearly interpolated onto A's wavelengths.
        append : bool, optional
          Append results to arrays, rather than creating new ones.

        Results
        -------
        self.wave : array of ndarray
          The wavelengths.
        self.spec : array of ndarray
          The spectra.
        self.var : array of ndarray
          The variance on the spectrum due to background.  If the
          background is not estimated, the variance will be based on
          the signal, integration time, gain, and read noise.
        self.profile : array of ndarray
          The spatial profiles.
        self.h : list of headers
          Meta data.

        """

        from mskpy import image

        N_peaks = len(self.peaks)
        if abcombine:
            if N_peaks != 2:
                raise ValueError(
                    "There must be two peaks when abcombine is requested.")
            print('Combining (sum) AB apertures.')

        if traces:
            trace = self.trace_fits
        else:
            trace = None

        profile = self._profile(im)
        if profile.ndim == 1:
            profile = profile[np.newaxis]

        if bgap is None:
            spec = image.spextract(im, self.peaks, rap, trace=trace,
                                   subsample=5)[1]
            if var is None:
                var = np.zeros_like(spec)
            else:
                var = image.spextract(var, self.peaks, rap, trace=trace,
                                      subsample=5)[1]
        else:
            n, spec, nbg, mbg, bgvar = image.spextract(
                im, self.peaks, rap, trace=trace, bgap=bgap,
                bgorder=bgorder, subsample=5)
            var = 2 * rap * bgvar * (2 * rap / nbg)

        if self.wavecal is None:
            # dummy wavelengths
            wave = np.tile(np.arange(im.shape[1], dtype=float), N_peaks)
            wave = wave.reshape((N_peaks, im.shape[1]))
        else:
            wave = image.spextract(self.wavecal, self.peaks, rap, mean=True,
                                   trace=trace, subsample=5)[1]

        if abcombine:
            w = wave[::2]
            s = spec[::2]
            v = var[::2]
            p = profile[::2]
            h_other = h[1::2]
            h = h[::2]
            for i in range(len(s)):
                h[i].add_history('AB beam combined (sum)')
                h[i]['ITOT'] = h[i]['ITIME'] + \
                    h_other[i]['ITIME'], 'Total integration time (sec)'

                b = i * 2 + 1

                mask = s[i].mask + wave[b].mask + spec[b].mask + var[b].mask

                x = np.interp(w[i], wave[b, ~mask], spec[b, ~mask])
                s[i] -= np.ma.MaskedArray(x, mask=mask)

                x = np.interp(w[i], wave[b, ~mask], var[b, ~mask])
                v[i] += np.ma.MaskedArray(x, mask=mask)

            wave = w
            spec = s
            var = v

        # for spextool compatability
        for i in range(len(h)):
            ps = h[i]['PLATE_SC']

            appos = ['{:.2f}'.format(x) for x in
                     (self.peaks - config['bottom']) * ps]
            h[i]['APPOSO01'] = appos[0], 'Aperture positions (arcsec) for order 01'
            if abcombine:
                h[i]['ABAPS'] = ','.join(
                    appos), 'Aperture positions before AB combination.'
            h[i]['AP01RAD'] = rap * ps, 'Aperture radius in arcseconds'
            h[i]['APREF'] = config['bottom'], 'Aperture position reference pixel'
            h[i]['BGORDER'] = bgorder, 'Background polynomial fit degree'
            if bgap is None:
                h[i]['BGSTART'] = 0
                h[i]['BGWIDTH'] = 0
            else:
                h[i]['BGSTART'] = bgap[0] * \
                    ps, 'Background start radius in arcseconds'
                h[i]['BGWIDTH'] = np.ptp(
                    bgap) * ps, 'Background width in arcseconds'
            h[i]['MODENAME'] = 'Prism', 'Spectroscopy mode'
            h[i]['NAPS'] = 1, 'Number of apertures'
            h[i]['NORDERS'] = 1, 'Number of orders'
            h[i]['ORDERS'] = '1', 'Order numbers'

            slit = [float(x) for x in h[i]['SLIT'].split('x')]
            h[i]['SLTW_ARC'] = slit[0]
            h[i]['SLTH_ARC'] = slit[1]
            h[i]['SLTW_PIX'] = slit[0] / ps
            h[i]['SLTH_PIX'] = slit[1] / ps

            h[i]['RP'] = int(82 * 0.8 / slit[0]), 'Resovling power'
            h[i]['DISP001'] = 0.00243624, 'Dispersion (um pixel-1) for order 01'
            h[i]['XUNITS'] = 'um', 'Units of the X axis'
            h[i]['YUNITS'] = 'DN / s', 'Units of the Y axis'
            h[i]['XTITLE'] = '!7k!5 (!7l!5m)', 'IDL X title'
            h[i]['YTITLE'] = 'f (!5DN s!u-1!N)', 'IDL Y title'

        if (self.spec is None) or (self.spec is not None and not append):
            self.wave = wave
            self.spec = spec
            self.var = var
            self.profile = p
            self.h = h
        else:
            self.wave = np.ma.concatenate((self.wave, wave))
            self.spec = np.ma.concatenate((self.spec, spec))
            self.var = np.ma.concatenate((self.var, var))
            self.profile = np.ma.concatenate((self.profile, profile))
            self.h.extend(h)

    def save_spectrum(self, wave, spec, var, h, filename, **kwargs):
        """Write an extracted spectrum to a FITS file.

        The file should be compatible with xspextool.

        """
        kwargs['output_verify'] = kwargs.get('output_verify', 'silentfix')
        x = np.c_[wave.filled(np.nan),
                  spec.filled(np.nan),
                  np.sqrt(var).filled(np.nan),
                  np.zeros(len(wave))].T
        j = np.flatnonzero(np.isfinite(np.prod(x, 0)))
        fits.writeto(filename, x[:, j], h, **kwargs)

    def save_extracted(self, fnformat='{data}-{n}', path='./', **kwargs):
        """Write all extracted spectra to FITS files.

        The file columns are wavelength, DN/s, and uncertainty.

        The files should be compatible with xspextool.

        Parameters
        ----------
        fnformat : string, optional
          The format string for file names.  Use `'{data}'` for the
          data type (spec, or profile).  Use `'{n}'` for the spectrum
          number which will be gleaned from the FITS headers.  '.fits'
          is always appended.
        path : string, optional
          Save to this directory.
        **kwargs
          `astropy.io.fits.writeto` keyword arguments.

        """

        from astropy.table import Table

        if self.spec is None:
            raise ValueError("No spectra have been extracted")

        for i in range(len(self.spec)):
            n = re.findall('.*-([0-9]+).[ab].fits', self.h[i]['IRAFNAME'],
                           re.IGNORECASE)[0]
            fn = os.sep.join((path, fnformat.format(data='spec', n=n)))
            self.save_spectrum(self.wave[i], self.spec[i], self.var[i],
                               self.h[i], fn, **kwargs)

            fn = os.sep.join((path, fnformat.format(data='profile', n=n)))
            tab = Table(data=[self.profile[i]], names=['profile'])
            tab.write(fn + '.csv', format='ascii.ecsv')
            print('Wrote spec and profile for observation {}'.format(n))

    def scales(self, spectra):
        """Derive relative scale factors for spectra."""
        from astropy.stats import sigma_clip

        scales = np.zeros(len(spectra))
        for i in range(len(spectra)):
            scales[i] = np.ma.median(sigma_clip(spectra[i]))

        scales = scales.max() / scales
        return scales
