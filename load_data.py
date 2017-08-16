from cPickle import load as cifar_load
from os.path import join as join_path
import numpy as np

CIFAR100_PATH = "<path-to-file>/cifar-100-python"
trainpath = join_path(CIFAR100_PATH, "train")
testpath = join_path(CIFAR100_PATH, "test")
metapath = join_path(CIFAR100_PATH, "meta")

def __getDict(path):
	fo = open(path, 'rb')
	dict = cifar_load(fo)
	fo.close()
	return dict

def __getData(dict, cindex):
	# get the data, category labels, and super category labels
	datalst = dict['data']
	finelablst = dict['fine_labels']
	coarselablst = dict['coarse_labels']

	# build the data and labels
	# note that labels are arbitrary numbers and need to be rescaled to between 0-4
	data = []
	labels = []

	for i in xrange( len(datalst) ):
		if coarselablst[i] == cindex: # check if supercategory is "food_containers"
			data.append( np.array( datalst[i] )/255.0 )
			labels.append( finelablst[i] )
	
	return data, labels

global unique
def __rescale(labels):
	global unique
	# rescale each label to 0-4
	unique = list( set(labels) )
	unique.sort()

	return [ unique.index(lab) for lab in labels ]

def __one_hot(size, labels):
	# convert each label to a one-hot vector
	retlabs = []
	for lab in labels:
		hot_vec = [0]*size
		hot_vec[lab] = 1
		retlabs.append(hot_vec)
	return retlabs

def __to_rgb(data):
	output = []
	for i in xrange( len(data) ):
		r = data[i][0:32*32]
		g = data[i][32*32:32*32*2]
		b = data[i][32*32*2:]
		output.append( np.reshape(zip(r, g, b), (32, 32, 3)) )
	return output

def __getNextBatch(obj, how_many):
	if obj.__current == -1 or obj.__current >= obj.num_examples:
		import random
		obj.__current = 0
		pairs = zip(obj.images, obj.labels)
		random.shuffle(pairs)
		obj.images, obj.labels = zip( *pairs )
	
	start = obj.__current
	obj.__current += how_many
	return obj.images[start:start+how_many], obj.labels[start:start+how_many]

def LoadData(img_vectorized=True, one_hot=False):
	traindict = __getDict(trainpath)
	testdict = __getDict(testpath)
	metadict = __getDict(metapath)

	category = "food_containers"
	categoryidx = metadict['coarse_label_names'].index(category)
	traindata, trainlabels = __getData(traindict, categoryidx)
	testdata, testlabels = __getData(testdict, categoryidx)

	trainlabels = __rescale(trainlabels)
	testlabels = __rescale(testlabels)

	class Object(object):
		pass

	train = Object()
	train.images = traindata if img_vectorized else __to_rgb(traindata)
	train.labels = __one_hot(max(trainlabels)+1, trainlabels) if one_hot else trainlabels
	train.__current = -1
	train.next_batch = lambda x: __getNextBatch(train, x)
	train.num_examples = len(trainlabels)

	test = Object()
	test.images = testdata if img_vectorized else __to_rgb(testdata)
	test.labels = __one_hot(max(testlabels)+1, testlabels) if one_hot else testlabels
	test.__current = -1
	test.next_batch = lambda x: __getNextBatch(test, x)
	test.num_examples = len(testlabels)

	cifar = Object()
	cifar.train = train
	cifar.test = test
	# cifar.names = namemaps

	return cifar
