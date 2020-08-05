import cPickle
import pprint

# filename = "../merge.en.downsample"
filename = "util/merge.en.output_cleaned"
# filename = "buangggg"
examples = cPickle.load( open(filename, "rb") )

print("Total No. of instances:{}".format(len(examples)))
# output format
# locations, genders, ages, texts, ratings

tot = 0
d = {}
target_idxs = [0, 1, 2, 4]
for example in examples:
	for target_id in target_idxs:
		if target_id not in d:
			d[target_id] = {}
		target = example[target_id]
		if target not in d[target_id]:
			d[target_id][target] = 1
		else:
			d[target_id][target] += 1
			# if d[target] <10:
			# 	print(example)
	tot+=1
print(example)
print(tot)
pprint.pprint(d)