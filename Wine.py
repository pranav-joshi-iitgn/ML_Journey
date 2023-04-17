from sklearn.datasets import load_wine
from matplotlib.pyplot import *
from numpy import *
wine = load_wine()
data = wine.data
xMax = amax(data,axis=0)
xMin = amin(data,axis=0)
labels = wine.target
bins = 5
means = []
variances = []
N = []
def prob(x):
    guessP = 0.5
    guess = None
    for c in range(3):
        mu = means[c]
        v = means[c]
        p = exp(-((x-mu)**2) / (2*v))*Max
        p = product(p) * N[c]/sum(N)
        if p > guessP:
            guessP = p
            guess = c 
    return guess

for c in range(3):
    figure()
    X = data[labels==c]
    N.append(X.shape[0])
    mu = mean(X,axis=0)
    v = var(X,axis=0)
    means.append(mu)
    variances.append(v)
    for i in range(12):
        subplot(3,4,i+1)
        x = X[:,i]
        p,xvals = histogram(x,bins)
        xvals = xvals[:-1]
        binWidth = xvals[1]-xvals[0]
        p = p/(x.shape[0]*binWidth)
        bar(xvals+binWidth/2,p,binWidth,color="gray")
        x = linspace(xMin[i],xMax[i],100)
        Max = (2*pi*v[i])**(-0.5)
        p = exp(-((x-mu[i])**2) / (2*v[i]))*Max
        plot(x,p,"red")
        vlines([mu[i]],0,Max,['blue'])
        yticks([])
        xlim([xMin[i],xMax[i]])
    suptitle(c)
show()
correct = 0
incorrect = 0
for i in range(sum(N)):
    guess = prob(data[i])
    if(guess!=None):
        if(guess==labels[i]):
            correct += 1
        else:
            incorrect += 1
print(100*correct/(correct+incorrect))
print(N)