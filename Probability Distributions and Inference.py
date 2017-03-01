#MEHMET TUGRUL SAVRAN
#JANUARY 2017


#These inputs are said to be space-separated strings that represent an array
#e.g. "123 4 56 6 8" which is ultimately [123,4,56,6,8]
def parse(string):
	array = string.split()
	array = map(int,array)
	return array

#Mean, Mode, Median
def mean(array):
	samplesize = len(array)
	summ = sum(array)
	floatmean = float(summ)/samplesize
	roundedmean = round(floatmean,1)
	return roundedmean


def mode(array):
	tally = {}
	for element in array:
		if element not in tally:
			tally[element] = 0
		if element in tally:
			tally[element] += 1
	liste = []
	for key in tally:
		liste.append((key,tally[key]))
	liste = sorted(liste, key = lambda sth:sth[0])
	liste = sorted(liste, key = lambda sth:sth[1],reverse = True)
	return liste[0][0]

def median(array):
	array = sorted(array)
	length = len(array)
	if length%2 == 0:
		toplam = array[(length/2)-1] + array[length/2]
		return float(toplam)/2
	return round(array[length/2],1)

#Weighted Mean

def weightedMean(array,weights):
	samplesize = len(array)
	summ = 0 
	for i in range(samplesize):
		summ += array[i]*weights[i]
	weightSum = sum(weights)
	floatMean = float(summ)/weightSum
	return round(floatMean,1)






#Standard Deviation

def standardDeviation(array):
	average = mean(array)
	samplesize = len(array)
	squaredmeansum = 0 
	for element in array:
		squaredmeansum += (element - average)**2
	variance = float(squaredmeansum)/samplesize
	stDev = variance**0.5
	return stDev



#Quartiles

def quartileFinder(array):
	array = sorted(array)
	length = len(array)
	if length%2 == 0:
		LowerHalf = array[:(length/2)]
		UpperHalf = array[length/2:]
		Q1 = median(LowerHalf)
		Q2 = median(array)
		Q3 = median(UpperHalf)
	else:
		LowerHalf = array[:length/2]
		UpperHalf = array[(length/2)+1:]
		Q1 = median(LowerHalf)
		Q2 = median(array)
		Q3 = median(UpperHalf)
	return (int(Q1),int(Q2),int(Q3))




#Interquartile Range
#Given an array of integers and an array of frequencies of these integers, 
#Construct the actual array and find the interquartile range

def interQuartileFinder(array,frequencies):
	newarray = []
	for i in range(len(array)):
		newarray.append(frequencies[i]*[array[i]])
	finalarray = []
	for array in newarray:
		for i in array:
			finalarray.append(i)
	finalarray = sorted(finalarray)
	quartiles = quartileFinder(finalarray)
	interquartileRange = quartiles[2] - quartiles[0]
	return round(interquartileRange,1)

#Continuing with...

#Binomial Distribution
#Helper function to calculate n choose k 
def nchoosek(n,k):
	nominator = 1
	denominator = 1 
	for i in range(k,0,-1):
		nominator *= n 
		n -= 1
		denominator *= i 
	return nominator/denominator

def binomPMF(successrate,i,n):
	return nchoosek(n,i)*(successrate**i)*((1-successrate)**(n-i))

def binomCDF(successrate, atmostK, n):
	answer = 0 
	for i in range(atmostK):
		probabilitymassfunction = binomPMF(successrate,i,n)
		answer += probabilitymassfunction
	return answer

#Geometric distribution
#Below calculates failures until success, which is basically geometricPMF
def geometricPMF(p_success,trials):
	return p_success * (1-p_success)**(trials-1)
# p_success = 1.0/3
# print geometricPMF(p_success,0)

def geometricCDF(p_success,trials):
	answer = 0
	for i in range(1,trials+1):
		answer += geometricPMF(p_success,i)
		#print answer
	return answer
#print round(geometricCDF(1.0/3,5),3)




#Poisson Distribution
def factorial(k):
	answer = 1
	for i in range(1,k+1):
		answer *= i 
	return answer

def factorialrecursive(k):
	if k == 1:
		return 1 
	return k*factorialrecursive(k-1)

def poisson(rate,k):
	e = 2.718281828459045
	return (rate**k)*e**(-rate)/factorial(k)
#print round(poisson(2.5,5),3)


#Normal distribution

def stdnormalpdf(x):
	e = 2.718281828459045
	pi = 3.14159265359
	return (e**(-0.5*(x**2)))/((2*pi)**0.5)

def generalnormalpdf(x,mu,sigma):
	x = float(x-mu)/sigma
	return (1/float(sigma))*stdnormalpdf(x)

#print generalnormalpdf(1000,30,4)-generalnormalpdf(21,30,4)


# print "general mean is",49*205

# print 15*49

# print "required mean weight per box is", 9800/49.00

# print round(1-0.369441,4)


#Pearson Correlation Coefficient 

def pearsonCC(X,Y):
	#Assumes X and Y are not trivially equal (e.g. 1,1,1 & 1,1,1)
	mu_x = mean(X)
	mu_y = mean(Y)
	sigma_x = standardDeviation(X)
	sigma_y = standardDeviation(Y)
	nominator = 0 
	n = len(X)
	for i in range(n):
		nominator += (X[i]-mu_x)*(Y[i]-mu_y)
	return nominator/float(n*sigma_x*sigma_y)
# cc = pearsonCC([10,9.8,8,7.8,7.7,7,6,5,4,2],[200, 44, 32, 24, 22, 17, 15, 12, 8, 4])
# print cc

def spearmanRankCC(X,Y):
	sorted_X = sorted(X)
	sorted_Y = sorted(Y)
	N = len(X)
	sigma = 0 
	for i in range(N):
		r_x = sorted_X.index(X[i])
		r_y = sorted_Y.index(Y[i])
		d_i = r_x - r_y
		sigma += d_i**2
	nominator = 6*sigma
	fraction = float(nominator)/(N**3-N)
	return 1-fraction
# print spearmanRankCC([10, 9.8, 8, 7.8, 7.7, 1.7, 6, 5, 1.4, 2],
					# [200, 44, 32, 24, 22, 17, 15, 12, 8, 4])

#Least Square Regression Line

def leastSquareRegressionLine(X,Y):

	"""Returns a tuple (a,b) where a is y-intercept and b
	is the slope of the line """

	'''Remember, regression line is given by Y = a + b*X'''

	#calculating b (b = rho*stdy/stdx)
	rho = pearsonCC(X,Y)
	sigma_y = standardDeviation(Y)
	sigma_x = standardDeviation(X)
	b = rho*float(sigma_y)/sigma_x

	#calculating a (a = mean(y) - b*mean(x))
	a = mean(Y) - b*mean(X)
	return (a,b)

def SST(Y):
	"""Returns Total Sums of Squares"""
	summation = 0
	average = mean(Y)
	for i in range(len(Y)):
		summation += (Y[i]-average)**2
	return summation

# X = [1,2,3,4,5]
# Y = [2,1,4,3,5]

# print leastSquareRegressionLine(X,Y)

#Sample Case. Below arrays denote students' math and statistics score
#We find a regression between them
# X = [95,85,80,70,60] #Math scores
# Y = [85,95,70,65,70] #Statistics scores

# equation = leastSquareRegressionLine(X,Y)
# y = equation[1]*80 + equation[0]
# print y 





