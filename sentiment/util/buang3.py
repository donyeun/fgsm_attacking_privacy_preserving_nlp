import _pickle as cPickle

output_data = [[1,2,3], [4,5,6]]


f = open('merge.en.output', 'wb')
cPickle.dump(output_data, f, protocol=0)
f.close()