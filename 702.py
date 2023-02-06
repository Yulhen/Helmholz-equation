import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import scipy.special as spe
from scipy import linalg


L = 0.0014
a = 0.0005 
n = 281

lam = 1e-4
k = 2*np.pi/lam
n0 = 3.45
nu = -1e-2
mu = -0.0054908455
delt =  0.0002
smax = 2


X, h = np.linspace(-L,L,n,retstep=True)

print(h)


p =  1.54158156 - 0.1845093486j
q = 8.3979819858 + 0.2887822041j

bettoch = -649.93 - 44.751j


def toch(x):
	if (abs(x) < a):
		return 0.5*(np.exp(1j*p*x/a)+np.exp(-1j*p*x/a))
	if (abs(x) > a):
		return (np.exp(1j*q*abs(x)/a))*np.cos(p)/(np.exp(1j*q))
			
def sig(x):
	if (x < -L+delt):
		return 1+1j*smax*np.power(((x+L-delt)/delt),2)
	if (x > L-delt):
		return 1+1j*smax*np.power(((x-L+delt)/delt),2)
	else:
		return 1.	
		
def nun(x):
	if (abs(x) < a):
		return nu
	else:
		return 0
		

def mum(x):
	if (abs(x) < a):
		return mu
	else:
		return 0

def u(x):
	B = np.zeros((len(X)-2,len(X)-2),dtype = complex)
	for i in range(1,len(x)-3):
		if (i!=0) and (i!=len(x)):
			ss = sig(x[i+1])
			sm = sig(x[i+1]-0.5*h)
			sp = sig(x[i+1]+0.5*h)
			nuu = nun(x[i+1])
			muu = mum(x[i+1])
			B[i][i-1] = 0.5/(ss*sm*k*n0*np.power(h,2))
			B[i][i+1] = 0.5/(ss*sp*k*n0*np.power(h,2))
			B[i][i] = -0.5*(1/(sm) + 1/(sp))/(ss*k*n0*np.power(h,2)) + 0.5*k*(2*n0*nuu + 1j*muu)/(n0)
			
			
	ss = sig(x[1])
	sm = sig(x[1]-0.5*h)
	sp = sig(x[1]+0.5*h)
	B[0][0] = -0.5*(1/(sm) + 1/(sp))/(ss*k*n0*np.power(h,2)) 
	B[0][1] = 0.5/(ss*sp*k*n0*np.power(h,2))
	
	
	ss = sig(x[-2])
	sm = sig(x[-2]-0.5*h)
	sp = sig(x[-2]+0.5*h)
	B[-1][-1] = -0.5*(1/(sm) + 1/(sp))/(ss*k*n0*np.power(h,2))
	B[-1][-2] = 0.5/(ss*sm*k*n0*np.power(h,2))
	
		
		
	return B   




def max2(z):
    s = 0
    b = 0
    for x in range(len(z)):
        if z[x] > s:
            s = z[x]
            b = x
    return b


B = u(X)
U = np.zeros((len(X),len(X)-2), dtype = complex)
bett, U1 = linalg.eig(B)
U[:][1:-1] = U1
g = -np.imag(bett)
#for i in range(len(X)-2):
#	print(B[i])
maxval = max2(g)
U = U*np.sqrt(n/4)
for k in range(len(bett)):
	U[:,k] = U[:,k]/max(U[:,k]) 
print(maxval, '    ', max(g))



tchr = np.zeros(len(X), dtype = complex)   
for j in range(len(X)):
	tchr[j] = toch(X[j])



plt.style.use('dark_background')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

# animate the time data
print(bett[244])

x = np.imag(U[:,244])
y = np.imag(tchr)

plt.plot(X,x,color='cyan')
plt.plot(X,y,color='red')
#plt.xlabel(erg)
plt.grid(True)
    

plt.show()
