import numpy as np 
import matplotlib.pyplot as plt

def PlotFFT(image, plot3D = True, realFFT = False, plotPhase = True, scaleAmplitude = True, xlim = None, ylim=None):

  # Spectrum calculation
  if realFFT:
    # FFT Shift only over height axis if real FFT
    spectrum = np.fft.fftshift(np.fft.rfft2(image,axes=(- 2, - 1)),axes = [-2])
    inverse_DFT = np.fft.irfft2(np.fft.ifftshift(spectrum,axes = [-2]))
    # axis labels for spectrum to allign with shifted signal
    extent = [
      0,np.shape(spectrum)[-1],
      int(np.shape(spectrum)[-2]/2),-int(np.shape(spectrum)[-2]/2)]
  else:
    spectrum = np.fft.fftshift(np.fft.fft2(image,axes=(- 2, - 1)))
    #spectrum = tf.signal.fftshift(tf.signal.fft2d(image))
    inverse_DFT = np.fft.ifft2(np.fft.ifftshift(spectrum)).real
    extent = [
      -int(np.shape(spectrum)[-1]/2),int(np.shape(spectrum)[-1]/2),
      int(np.shape(spectrum)[-2]/2),-int(np.shape(spectrum)[-2]/2)]
    
  amplitude_spectrum = np.abs(spectrum)**2 # amplitudes of complex spectrum

  if scaleAmplitude:
    amplitude_spectrum = np.log(amplitude_spectrum)

  if plotPhase:
    phase_spectrum = np.angle(spectrum)
    columns = 4
  else:
    columns = 3

  #3D Plot preparation
  if plot3D:
      rows = 2
  else:
      rows = 1

  # Plotting
  fig = plt.figure(figsize=[columns*8,6*rows])
  label_fontsize = 16
  title_fontsize = 20

  ax = fig.add_subplot(rows, columns, 1)
  ax.set_title('Image',fontsize=title_fontsize)
  plt.imshow(image, cmap='gray')
  plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
  plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
  clb = plt.colorbar()
  clb.ax.tick_params(labelsize=label_fontsize)

  ax = fig.add_subplot(rows, columns, 2)
  ax.set_title('2D DFT Amplitude',fontsize=title_fontsize)
  plt.imshow(amplitude_spectrum, cmap='jet',extent=extent)
  plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
  plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
  if xlim:
    plt.xlim(xlim)
  if ylim:
    plt.ylim(ylim)
  clb = plt.colorbar()
  clb.ax.tick_params(labelsize=label_fontsize)

  if plotPhase:  
    ax = fig.add_subplot(rows, columns, 3)
    ax.set_title('2D DFT Phase',fontsize=title_fontsize)
    plt.imshow(phase_spectrum, cmap='jet',extent=extent)
    plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
    plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
    if xlim:
      plt.xlim(xlim)
    if ylim:
      plt.ylim(ylim)
    clb = plt.colorbar()
    clb.ax.tick_params(labelsize=label_fontsize)

  ax = fig.add_subplot(rows, columns, columns)
  ax.set_title('2D iDFT',fontsize=title_fontsize)
  plt.imshow(inverse_DFT, cmap='gray')
  plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
  plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
  clb = plt.colorbar()
  clb.ax.tick_params(labelsize=label_fontsize)
    
  if plot3D:
      X, Y = np.meshgrid(range(0,np.shape(image)[0]), range(0,np.shape(image)[1]))
      if realFFT:
        X_freq, Y_freq = np.meshgrid(range(0,np.shape(spectrum)[-2]), range(-int(np.shape(spectrum)[-1]/2),int(np.shape(spectrum)[-1]/2)))
      else:
        X_freq, Y_freq = np.meshgrid(range(-int(np.shape(spectrum)[-2]/2),int(np.shape(spectrum)[-2]/2)), range(-int(np.shape(spectrum)[-1]/2),int(np.shape(spectrum)[-1]/2)))

      ax = fig.add_subplot(2, columns, columns+1,projection='3d')
      ax.set_title('Image',fontsize=title_fontsize)
      ax.plot_surface(X,Y,image,cmap='gray')
      plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
      plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
      plt.setp(ax.get_zticklabels(),fontsize=label_fontsize)

      ax = fig.add_subplot(2, columns, columns+2,projection='3d')
      ax.set_title('2D DFT Amplitude',fontsize=title_fontsize)
      ax.plot_surface(X_freq,Y_freq,amplitude_spectrum,cmap='jet')
      plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
      plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
      plt.setp(ax.get_zticklabels(),fontsize=label_fontsize)

      if plotPhase:
        ax = fig.add_subplot(2, columns, columns+3,projection='3d')
        ax.set_title('2D DFT Phase',fontsize=title_fontsize)
        ax.plot_surface(X_freq,Y_freq,phase_spectrum,cmap='jet')
        plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
        plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
        plt.setp(ax.get_zticklabels(),fontsize=label_fontsize)

      ax = fig.add_subplot(2, columns, 2*columns,projection='3d')
      ax.set_title('2D iDFT',fontsize=title_fontsize)
      ax.plot_surface(X,Y,inverse_DFT,cmap='gray')
      plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
      plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
      plt.setp(ax.get_zticklabels(),fontsize=label_fontsize)

  plt.show()

# === 
def PlotFFTWithFilter(image, filter, xlim = None, ylim=None, cropToMask = False, shiftFFT = True, scaleAmplitude = True, plotPhase = True):

  # Spectrum calculation
  if shiftFFT:
    spectrum = np.fft.fftshift(np.fft.fft2(image,axes=(- 2, - 1)))
    inverse_DFT = np.fft.ifft2(np.fft.ifftshift(spectrum)).real
  else:
    spectrum = np.fft.fft2(image,axes=(- 2, - 1))
    inverse_DFT = np.fft.ifft2(spectrum).real
  amplitude_spectrum = np.abs(spectrum)**2 # amplitudes of complex spectrum
  if scaleAmplitude:
    amplitude_spectrum = np.log(amplitude_spectrum)
  if plotPhase:
    phase_spectrum = np.angle(spectrum)
    columns = 4
  else:
    columns = 3
  extent = [
      -int(np.shape(spectrum)[-1]/2),int(np.shape(spectrum)[-1]/2),
      int(np.shape(spectrum)[-2]/2),-int(np.shape(spectrum)[-2]/2)]

  filtered_spectrum = spectrum * filter
  if cropToMask:
    true_points = np.argwhere(filter)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    filtered_spectrum = filtered_spectrum[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1]    
    
  filtered_extent = [
      -int(np.shape(filtered_spectrum)[-1]/2),int(np.shape(filtered_spectrum)[-1]/2),
      int(np.shape(filtered_spectrum)[-2]/2),-int(np.shape(filtered_spectrum)[-2]/2)]
  filtered_amplitude_spectrum = np.abs(filtered_spectrum)**2
  if scaleAmplitude:
    filtered_amplitude_spectrum=np.log(filtered_amplitude_spectrum)
  if plotPhase:
    filtered_phase_spectrum = np.angle(filtered_spectrum)
  if shiftFFT:
    filtered_inverse_DFT = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)).real
  else:
    filtered_inverse_DFT = np.fft.ifft2(filtered_spectrum).real

  # Plotting
  fig = plt.figure(figsize=[columns*8,12])
  label_fontsize = 16
  title_fontsize = 20

  ax = fig.add_subplot(2, columns, 1)
  ax.set_title('Image',fontsize=title_fontsize)
  plt.imshow(image, cmap='gray')
  plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
  plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
  clb = plt.colorbar()
  clb.ax.tick_params(labelsize=label_fontsize)

  ax = fig.add_subplot(2, columns, 2)
  ax.set_title('2D DFT Amplitude',fontsize=title_fontsize)
  plt.imshow(amplitude_spectrum, cmap='jet',extent=extent)
  plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
  plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
  if xlim:
    plt.xlim(xlim)
  if ylim:
    plt.ylim(ylim)
  clb = plt.colorbar()
  clb.ax.tick_params(labelsize=label_fontsize)

  if plotPhase:
    ax = fig.add_subplot(2, columns, 3)
    ax.set_title('2D DFT Phase',fontsize=title_fontsize)
    plt.imshow(phase_spectrum, cmap='jet',extent=extent)
    plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
    plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
    if xlim:
      plt.xlim(xlim)
    if ylim:
      plt.ylim(ylim)
    clb = plt.colorbar()
    clb.ax.tick_params(labelsize=label_fontsize)

  ax = fig.add_subplot(2, columns, columns)
  ax.set_title('2D iDFT',fontsize=title_fontsize)
  plt.imshow(inverse_DFT, cmap='gray', vmin=0, vmax=255)
  plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
  plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
  clb = plt.colorbar()
  clb.ax.tick_params(labelsize=label_fontsize)

  ax = fig.add_subplot(2, columns, columns+1)
  ax.set_title('Filter F-Domain',fontsize=title_fontsize)
  plt.imshow(filter, cmap='gray')
  plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
  plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
  clb = plt.colorbar()
  clb.ax.tick_params(labelsize=label_fontsize)

  ax = fig.add_subplot(2, columns, columns+2)
  ax.set_title('Filtered Amplitude Spectrum',fontsize=title_fontsize)
  plt.imshow(filtered_amplitude_spectrum, cmap='jet',extent=filtered_extent)
  plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
  plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
  if xlim:
    plt.xlim(xlim)
  if ylim:
    plt.ylim(ylim)
  clb = plt.colorbar()
  clb.ax.tick_params(labelsize=label_fontsize)

  if plotPhase:
    ax = fig.add_subplot(2, columns, columns+3)
    ax.set_title('Filtered Phase Spectrum',fontsize=title_fontsize)
    plt.imshow(filtered_phase_spectrum, cmap='jet',extent=filtered_extent)
    plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
    plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
    if xlim:
      plt.xlim(xlim)
    if ylim:
      plt.ylim(ylim)
    clb = plt.colorbar()
    clb.ax.tick_params(labelsize=label_fontsize)

  ax = fig.add_subplot(2, columns, 2*columns)
  ax.set_title('Filtered iDFT',fontsize=title_fontsize)
  plt.imshow(filtered_inverse_DFT, cmap='gray', vmin=0, vmax=255)
  plt.setp(ax.get_xticklabels(),fontsize=label_fontsize)
  plt.setp(ax.get_yticklabels(),fontsize=label_fontsize)
  clb = plt.colorbar()
  clb.ax.tick_params(labelsize=label_fontsize)

  plt.show()