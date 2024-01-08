T = 1
f1 = 2
f2 = 5
from matplotlib.pyplot import *
from numpy import *
import scipy as sp
from scipy import fft
n =100
s = 10
x = linspace(0,T,n)
I = []
for y in linspace(0,T,n):
  I.append(2*sin(2*pi*(f1*x+f2*y)) + 5*sin(2*pi*(f2*x+f1*y)))
imshow(I,cmap="gray")
title("z as a function of x and y")
Ticks = 5
xticks(linspace(0,n,Ticks),linspace(0,T,Ticks))
yticks(linspace(0,n,Ticks),linspace(0,T,Ticks))
ax = figure().add_subplot(projection='3d')
FT = sp.fft.fft2(I)
I = 2*abs(FT[:s,:s])/(n*n)
fx = arange(s)/T
fx = tile(fx,(s,1))
fy = (arange(s)/T)
fy = tile(fy,(s,1)).T
ax.plot_surface(fx,fy,I)
title("Fourier Transform of z")
figure()
C = sp.fft.ifft2(FT) 
I = real(C)
imshow(I,cmap="gray")
title("Inverse FT of FT of z")
xticks(linspace(0,n,Ticks),linspace(0,T,Ticks))
yticks(linspace(0,n,Ticks),linspace(0,T,Ticks))
show()