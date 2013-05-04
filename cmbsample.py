# Sample a 2d reaslization based on a power spectrum and
# noisy data.
import numpy as np
from matplotlib.pylab import *
from enutil import CG

def Sfun(l): return 0.001/(l+1e-3)**3
n   = 0x200
ext = np.array((n,n))
xpos = np.array(np.meshgrid(np.arange(ext[1]),np.arange(ext[0])),dtype=float)[::-1]
lpos = np.array(np.meshgrid(np.fft.fftfreq(ext[1]),np.fft.fftfreq(ext[0])))[::-1]

S = Sfun(np.sum(lpos**2,0)**0.5)
N = (np.sum((xpos-ext[:,None,None]/2)**2,0)/np.sum(ext**2)+1e-1)*3
M = np.sum((xpos-ext[:,None,None]/2)**2,0)**0.5 > n/4

s = np.real(np.fft.ifft2(np.fft.fft2(np.random.standard_normal(ext))*S**0.5,ext))
n = np.random.standard_normal(ext)*N**0.5

d = (s+n)*M

# Mark mask as totally uncertain
N /= M

# Will now solve [N"+S"]x = N"d + N"u + N"v by conjugate gradients
def A(x): return x/N + np.real(np.fft.ifft2(np.fft.fft2(x)/S,ext))
def Ap(x): return np.real(np.fft.ifft2(np.fft.fft2(x)*S,ext))
def dot(a,b): return np.sum(a*np.conj(b))
u = np.random.standard_normal(ext)
v = np.random.standard_normal(ext)
b = d/N + u/N**0.5 + np.real(np.fft.ifft2(np.fft.fft2(v)/S**0.5,ext))

cg = CG(A, b, x0=d, M=Ap, dot=dot)
while True:
	cg.step()
	print "%3d %15.7e %15.7e" % (cg.i, cg.err, cg.err_true)
	if cg.i % 0x10 == 0:
		matshow(cg.x)
		colorbar()
		show()
