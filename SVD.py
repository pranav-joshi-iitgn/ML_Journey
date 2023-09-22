from matplotlib.pyplot import *
from numpy import *
from PIL import Image
image = Image.open('/home/hp/Pictures/Roll.png').convert('L')
M = asarray(image)
print(M.shape)
#M = M[:500,:500]
u,s,vh = linalg.svd(M)
print(s.shape)
print(list(map(int,s)))
Grad = s[0]*(array([u[:,0]]).T @ [vh[0]]) 
Mnew = Grad*0
F = figure()
G = F.add_subplot(121)
G.set_title("ith term")
MN = F.add_subplot(122)
MN.set_title("Sum of first i terms")
g = G.imshow(Mnew,cmap="gray")
mn = MN.imshow(Mnew,cmap="gray")
for i in range(50):
    pause(1/2-arctan(i/10-1)/pi)
    Grad = s[i]*(array([u[:,i]]).T @ [vh[i]]) 
    Mnew += Grad
    g.set_data(Grad)
    g.autoscale()
    mn.set_data(Mnew)
    mn.autoscale()
    draw()
show()