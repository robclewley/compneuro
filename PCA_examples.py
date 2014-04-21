from PyDSTool import *

def princomp(A):
    """ performs principal components analysis
    (PCA) on the n-by-p data matrix A
    Rows of A correspond to observations, columns to variables.

    Returns :
     coeff :
    is a p-by-p matrix, each column containing coefficients
    for one principal component.
     score :
    the principal component scores; that is, the representation
    of A in the principal component space. Rows of SCORE
    correspond to observations, columns to components.
     latent :
    a vector containing the eigenvalues
    of the covariance matrix of A.

    Taken from: http://glowingpython.blogspot.com/2011/07/principal-component-analysis-with-numpy.html
    """
    # computing eigenvalues and eigenvectors of covariance matrix
    # subtract the mean (along columns)
    M = (A-mean(A.T,axis=1)).T
    # attention:not always sorted
    [latent,coeff] = linalg.eig(cov(M))

    # projection of the data in the new space
    score = dot(coeff.T,M)
    return coeff,score,latent

def annotate(ax, name, start, end):
    arrow = ax.annotate(name,
                        xy=end, xycoords='data',
                        xytext=start, textcoords='data',
                        arrowprops=dict(facecolor='red', width=2.0))
    return arrow

# Surrogate data
N = 1000
wTrue = np.linspace(0, 1000, N)

yTrue = 20*np.sin(wTrue/100.0) + 0.1 * wTrue
xTrue = 3 * wTrue + 10*yTrue

wData = wTrue + np.random.normal(0, 100, N)
xData = xTrue + np.random.normal(0, 100, N)

# Add to dimensions with uncorrelated random values
yData = yTrue + np.random.normal(0, 6, N)
#yData = np.random.uniform(-10, 10, N)
zData = np.random.uniform(-20, 20, N)

wData = np.reshape(wData, (N, 1))
xData = np.reshape(xData, (N, 1))
yData = np.reshape(yData, (N, 1))
zData = np.reshape(zData, (N, 1))

data = np.hstack((wData, xData, yData, zData))
mu = mean(data, axis=0)

def alt_SVD(data):
    data = data - mu
    # data = (data - mu)/data.std(axis=0)  # Uncomment this reproduces mlab.PCA results
    eigenvectors, eigenvalues, V = np.linalg.svd(
        data.T, full_matrices=False)
    projected_data = np.dot(data, eigenvectors)
    sigma = projected_data.std(axis=0).mean()
    ##print(eigenvectors)
    return eigenvectors, eigenvalues, projected_data, sigma

##fig, ax = plt.subplots()
##ax.scatter(xData, yData)
##ax.set_aspect('equal')
##for axis in eigenvectors:
##    annotate(ax, '', mu, mu + sigma * axis)

coeff, score, latent = princomp(data)
perc = cumsum(latent)/sum(latent)

kscale = 300

plt.figure(1)
plt.subplot(121)
# every eigenvector describe the direction
# of a principal component.
ax = plt.gca()
plt.plot(data.T[0,:], data.T[1,:],'ob') # the data
plt.plot([0, -coeff[0,0]*kscale]+mu[0], [0, -coeff[0,1]*kscale]+mu[1],'-k', lw=3)
plt.plot([0, coeff[1,0]*kscale]+mu[0], [0, coeff[1,1]*kscale]+mu[1],'-k', lw=3)

eigenvectors, eigenvalues, proj_data, sigma = alt_SVD(data)
#proj_data = np.dot(data, coeff)
#sigma = proj_data.std(axis=0).mean()
#for eigvec in coeff:
#    annotate(ax, '', mu, mu+sigma*eigvec)
for eigvec in eigenvectors[0:2]:
    annotate(ax, '', mu[0:2], mu[0:2]+sigma*eigvec[0:2])


ax.set_aspect('equal') #plt.axis('equal')
plt.subplot(122)
# projected data
plt.plot(score[0,:],score[1,:],'*g')
plt.axis('equal')

1/0

plt.figure(2)
plt.subplot(121)
ax = plt.gca()
# every eigenvector describe the direction
# of a principal component.
plt.plot(data.T[2,:],data.T[3,:],'ob') # the data
plt.plot([0, -coeff[0,2]*kscale]+mu[2], [0, -coeff[0,3]*kscale]+mu[3],'-k', lw=3)
plt.plot([0, coeff[1,2]*kscale]+mu[2], [0, coeff[1,3]*kscale]+mu[3],'-k', lw=3)

for eigvec in eigenvectors[2:4]:
    annotate(ax, '', mu[2:4], mu[2:4]+sigma*eigvec[2:4])

plt.axis('equal')
plt.subplot(122)
# projected data
plt.plot(score[2,:],score[3,:],'*g')
plt.axis('equal')


plt.show()