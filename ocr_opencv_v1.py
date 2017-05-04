import numpy as np
import cv2
from matplotlib import pyplot as plt



def GaussianFilter(sigma):
    halfSize = 3 * sigma
    maskSize = 2 * halfSize + 1 
    mat = np.ones((maskSize,maskSize)) / (float)( 2 * np.pi * (sigma**2))
    xyRange = np.arange(-halfSize, halfSize+1)
    xx, yy = np.meshgrid(xyRange, xyRange)    
    x2y2 = (xx**2 + yy**2)    
    exp_part = np.exp(-(x2y2/(2.0*(sigma**2))))
    mat = mat * exp_part

    return mat


def findAccuracy(cells, thisTest):


	preProccess1Test = np.array(thisTest)
	finalTest = preProccess1Test.reshape(-1, 784).astype(np.float32)

	#print preProccess1Test.shape
	#print finalTest.shape

	# Make it into a Numpy array. It size will be (50,100,20,20)
	npArrayInput = np.array(cells)

	# Now we prepare train_data and test_data.
	train = npArrayInput[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
	test = npArrayInput[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

	# Create labels for train and test data
	k = np.arange(10)
	train_labels = np.repeat(k,250)[:,np.newaxis]
	test_labels = train_labels.copy()

	# Initiate kNN, train the data, then test it with test data for k=1
	knn = cv2.KNearest()
	knn.train(train,train_labels)
	ret,result,neighbours,dist = knn.find_nearest(test, k=5)

	# Now we check the accuracy of classification
	# For that, compare the result with test_labels and check which are wrong
	matches = result==test_labels
	correct = np.count_nonzero(matches)
	accuracy = correct*100.0/result.size
	print accuracy






img1 = cv2.imread('digits.png')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('digits_grid.png')
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

gaussianFilter = GaussianFilter(1)
gaussianGray1 = cv2.filter2D(gray1, -1, gaussianFilter)
gaussianGray2 = cv2.filter2D(gray2, -1, gaussianFilter)


# Now we split the image to 5000 cells, each 20x20 size
cells1 = [np.hsplit(row,100) for row in np.vsplit(gaussianGray1,50)]

cells2 = [np.hsplit(row,25) for row in np.vsplit(gaussianGray2,20)]

findAccuracy(cells1, cells2)


