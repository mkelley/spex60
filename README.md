# spex60
IRTF SpeX instrument reduction for the 60 arcsec slit in low-resolution mode.

## Nominal example

1. Process calibration (flats, arcs) data.  Files will automatically
   be grouped and results saved to the directory "cal-YYYYMMDD":

	```python
	from glob import glob
	from spex60 import Prism60
	
	spex = Prism60()
	spex.process_cal(glob('bigdog/flat*.fits'))
	spex.process_cal(glob('bigdog/arc*.fits'))
	```

2. Reduce telluric calibrator.

	```python
	import os
	from glob import glob
	from spex60 import Prism60

	# generally, file lists need to be sorted (allows for ABBA sequence finding)
	files = sorted(glob('bigdog/spec*.fits'))

	spex = Prism60()
	spex.load_flat('cal-20151229/flat-00029-00033.fits')
	spex.load_wavecal('cal-20151229/wavecal-00034.fits')
	stack, var, headers = spex.read_numbered(files, between=[19, 28], pair=True)
	
	# median combine
	im, var, header = spex.median_combine(stack, variances, headers, axis=0)
	
	# save image
	if not os.path.exists('rx-20151229'):
		os.mkdir('rx-20151229')

	spex.save_image(im, var, header, 'rx-20151229/hd3266a.fits')
	```

3. Extract spectrum of telluric calibrator:

	```python
	rap = 30
	bgap = 50, 100
	
	spex.peak(im, ex_rap=rap, bgap=bgap)
	spex.trace(im)
	spex.extract(im, header, rap, bgap=bgap)
	```
	
4. Reduce target images:

	```python
	spex.load_flat('cal-20151229/flat-00013-00017.fits')
	spex.load_wavecal('cal-20151229/wavecal-00018.fits')
	stack, variances, headers = spex.read_numbered(
		files, between=[3, 12], pair=True)
	im, var, header = spex.median_combine(stack, variances, headers, axis=0)
	spex.save_image(im, var, header, 'rx-20151229/c2013x1.fits')
	```
	
5. Check the aperture definitions for the object (and revise as needed):

	```python
	bgap = 100, 130
	spex.peak(stack[0], ex_rap=rap, bgap=bgap)
	```
	
5. Extract target spectra, file-by-file, using the trace from the
   telluric standard.  Append extracted spectra to the SpeX data
   object, keeping the already extracted telluric standard:

	```python
	for i in range(len(stack)):
		spex.peak(stack[i], ex_rap=rap, bgap=bgap, plot=False)
		spex.extract(stack[i], headers[i], rap, var=variances[i], bgap=bgap, append=True)
	```

6. Plot raw spectra:

	```python
	import matplotlib.pyplot as plt
	
	fig = plt.figure()
	for i in range(len(spex.spec)):
		plt.plot(spex.wave[i], spex.spec[i])
	plt.yscale('log')
	```

7. Save profiles and raw spectra:

	```python
    if not os.path.exists('spex-20151229'):
        os.mkdir('spec-20151229')
    spex.save_extracted(path='spec-20151229')
	```

8. Combine target spectra, plot, and save:

  ```python
  import numpy as np
  wave = np.ma.mean(spex.wave[1:], 0)
  spec, var, h = spex.median_combine(spex.spec[1:], spex.var[1:], spex.h[1:])
  plt.plot(wave, spec, color='k')
  ```
