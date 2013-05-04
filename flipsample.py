# Flipper version of cmbsample. Needs this to use actual physical units
import numpy as np, scipy.interpolate, argparse, time
import flipper, flipperPol, fortran
import pyfftw.interfaces.numpy_fft
from matplotlib.pylab import *
from enutil.cg import CG
fft = pyfftw.interfaces.numpy_fft
#fft = np.fft

def makegrid(dims):
	res = np.empty([len(dims)]+[len(d) for d in dims])
	for i, d in enumerate(dims):
		islice = [None]*len(dims)
		islice[i] = Ellipsis
		res[i] = d[islice]
	return res

class ArrInterpol:
	def __init__(self, x, y, dim=-1):
		self.dim   = y.ndim+dim
		self.shape = y.shape
		self.y     = np.rollaxis(y, dim,y.ndim)
		self.y     = np.reshape(self.y, [np.prod(self.y.shape[:-1]),self.y.shape[-1]])
		self.x     = x
		self.inter = [scipy.interpolate.InterpolatedUnivariateSpline(x, yi) for yi in self.y]
	def __call__(self, x):
		res  = np.empty([self.y.shape[0]]+list(x.shape))
		low  = np.where(x < self.x[0])
		high = np.where(x > self.x[-1])
		ok   = np.where((x >= self.x[0])*(x <= self.x[-1]))
		for i in range(len(self.y)):
			res[i][ok]   = self.inter[i](x[ok])
			res[i][low]  = self.y[i,0]
			res[i][high] = self.y[i,-1]
		res = np.reshape(res, list(self.shape[:self.dim])+list(self.shape[self.dim+1:])+list(x.shape))
		for i in range(x.ndim):
			res = np.rollaxis(res, -x.ndim+i, self.dim-x.ndim+i-2)
		return res

class Flatsky:
	def __init__(self, patch, ncomp):
		self.patch = patch
		self.ncomp = ncomp
		self.x     = makegrid([np.arange(patch.Ny)*patch.pixScaleY,np.arange(patch.Nx)*patch.pixScaleX])
		self.l     = makegrid([fft.fftfreq(patch.Ny,d=patch.pixScaleY)*2*np.pi,fft.fftfreq(patch.Nx,d=patch.pixScaleX)*2*np.pi])
		self.absl  = np.sum(self.l**2,0)**0.5
		self.angl  = np.arctan2(self.l[1],self.l[0])
		self.shape = np.array(self.x.shape[1:])
	def fft(self, a):
		t1 = time.time()
		res = fft.fft2(a, axes=[-2,-1])
		print "fft", time.time()-t1
		return res
	def ifft(self, fa):
		t1 = time.time()
		res = np.real(fft.ifft2(fa, self.shape, axes=[-2,-1]))
		print "ifft", time.time()-t1
		return res
	def rand(self):
		return np.random.standard_normal([self.ncomp]+list(self.shape))
	def Cteb_to_Ctqu(self,C):
		if C.shape[0] == 1: return C
		cos2psi = np.cos(2*self.angl)
		sin2psi = np.sin(2*self.angl)
		rot = np.array([[cos2psi,-sin2psi],[sin2psi,cos2psi]])
		res = C.copy()
		res[1:,1:] = np.einsum("ijyx,jkyx->ikyx",rot,np.einsum("ijyx,kjyx->ikyx",C[1:,1:],rot))
		return res

def mmul(mat, vec):
	t1 = time.time()
	res = np.einsum("ijyx,jyx->iyx", mat, vec)
	print "mmul", time.time()-t1
	return res
def mpow(mat, exp):
	t1 = time.time()
	tmp = mat.reshape(mat.shape[0],mat.shape[1],np.prod(mat.shape[2:]))
	tmp = fortran.eigpow(tmp, exp)
	print "mpow", time.time()-t1
	return np.reshape(tmp,mat.shape)
	

parser = argparse.ArgumentParser()
parser.add_argument("noise_map")
parser.add_argument("power_spectrum")
args = parser.parse_args()

print "Reading input"
# Read the noise map (we haven't defined a proper format for polarized
# noise maps yet, so for now I will assume that T, Q and U have uncorrelated
# noise, which is the same for all components).
noisemap = flipper.liteMap.liteMapFromFits(args.noise_map)
# Read the power spectrum, which is just a text file with Dl-amplitudes,
# with ordering l, TT, TE, TB, EE, EB, BB
powspec  = np.loadtxt(args.power_spectrum).T
l, powspec = powspec[0], powspec[1:]
powspec *= 2*np.pi/l/(l+1)
powspec[np.isnan(powspec)] = 0
Cl = np.zeros((3,3,powspec.shape[-1]))
i = 0
for I in np.ndindex(Cl.shape[:-1]):
	if I[0] > I[1]:
		Cl[I] = Cl[I[::-1]]
	else:
		Cl[I] = powspec[i]
		i += 1

print "Interpolating spectra"
Csmooth = ArrInterpol(l, Cl)

# Set up our basis and matrices
sky = Flatsky(noisemap, Cl.shape[0])
iN  = np.zeros([sky.ncomp]*2+list(sky.shape))
print noisemap.pixScaleY*noisemap.pixScaleX
for i in range(sky.ncomp): iN[i,i,:] = noisemap.data*(noisemap.pixScaleY*noisemap.pixScaleX)
S   = Csmooth(sky.absl)/(noisemap.pixScaleY*noisemap.pixScaleX)
S   = sky.Cteb_to_Ctqu(S)
iS  = mpow(S, -1)

#iN /= 1000

# Generate a random map
print "Building realizations"
foo = mpow(S,0.5)
s   = sky.ifft(mmul(mpow(S,0.5),sky.fft(sky.rand())))
n   = mmul(mpow(iN,-0.5),sky.rand())
mask= iN[0,0] > np.max(iN[0])/1000
d   = s+n
n *= mask
d *= mask
iN *= mask

matshow(s[0])
colorbar()
matshow(s[1])
colorbar()
matshow(s[2])
colorbar()
show()

print "Building b"
def A(x): return mmul(iN,x) + sky.ifft(mmul(iS,sky.fft(x)))
def M(x): return sky.ifft(mmul(S,sky.fft(x)))
b = mmul(iN,d) + mmul(mpow(iN,0.5),sky.rand()) + sky.ifft(mmul(mpow(iS,0.5),sky.fft(sky.rand())))
def dot(a,b): return np.sum(a*b)

print "Solving"
cg = CG(A, b, x0=d, M=M, dot=dot)
while True:
	cg.step()
	print (cg.i, cg.err, cg.err_true)
	if cg.i % 10 == 0:
		matshow(cg.x[0])
		colorbar()
		matshow(cg.x[1])
		colorbar()
		matshow(cg.x[2])
		colorbar()
		show()
