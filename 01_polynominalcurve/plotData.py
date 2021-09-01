import matplotlib.pyplot as plt
def plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]):

    #plot every thing
    p1 = plt.plot(x1,t1,'bo')  #training data
    p2 = plt.plot(x2,t2,'g')  # true value
    if (x3 is not None):
        p3 = plt.plot(x3,t3,'r')
    
    #add title, legend and axes labels
    plt.ylabel('t')
    plt.xlabel('x')
    
    if (x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    else :
        plt.legend((p1[0],p2[0],p3[0]),legend)

import numpy as np
def fitdata(x,t,M):
    '''fitdata(x,t,M): Fit a polynominal of order M to the data (x,t) '''
    X = np.array([x**m for m in range(M+1)]).T
    w = np.linalg.inv(X.T@X)@X.T@t
    return w


#Load Data
beerData = np.loadtxt('/Users/mac/FML/LectureNotes/01_PolynomialCurveFitting/beer_foam.dat.txt')
plt.scatter(beerData[:,0],beerData[:,1],color = 'red')
plt.scatter(beerData[:,0],beerData[:,2],color = 'blue')
plt.scatter(beerData[:,0],beerData[:,3],color = 'orange')

# then we can fit the data using the polynominal curve fitting method we thrived
x = beerData[:,0]
t = beerData[:,1]
w = fitdata(x,t,M = 10)
print("w = :",w)  # w here are the parameters of the plolynominal curve model in order of 9

#                                                            #
# so in this function M = 9 , N = 15 , X belong to R(9+1*15) #
#                                                            #

#Now lets use the weight(w parameter 参数/权重)in test
xrange = np.arange(beerData[0,0],beerData[beerData.shape[0]-1,0],0.001)  #get equally spaced points in the xrange
X = np.array([xrange**m for m in range(w.size)]).T
esty = X@w #compute the predicted value

plotData(x,t,xrange,esty,legend=['Training Data','Estimated\nPlolynominal'])

#What will the foam height be at t = __?
# Initialize 't' as float so below type is correct
t_predict = np.float64(450)
x_test = np.array([t_predict**m for m in range(w.size)]).T
print("x_test = :",x_test)
predict_height = x_test@w
print("predict_height = :",predict_height)


