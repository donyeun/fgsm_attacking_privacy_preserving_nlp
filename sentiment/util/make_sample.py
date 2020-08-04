import os
import _pickle as cPickle
# import json
import ast
from tqdm import tqdm
from langid.langid import LanguageIdentifier, model
from sklearn.model_selection import train_test_split, ShuffleSplit
import pandas as pd
from sklearn.utils import shuffle

def is_english(text, lang_identifier, threshold=0.75):
	lang, prob = lang_identifier.classify(text)
	if lang == 'en':
		return True
	return False



DATASET_FOLDER = '../../dataset/external/www2015'
output_data = []

lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)


filenames = os.listdir(DATASET_FOLDER)
# filenames = ['denmark.auto-adjusted_gender.NUTS-regions.jsonl.tmp']
filenames = [
	'united_kingdom.auto-adjusted_gender.NUTS-regions.jsonl.tmp',
	'united_states.auto-adjusted_gender.geocoded.jsonl.tmp'
]
for filename in filenames:
	if filename.endswith('.jsonl.tmp'):
		error = 0
		valid = 0
		gender = 0
		loc = 0
		age = 0
		count = 0
		with open(os.path.join(DATASET_FOLDER, filename), 'r', errors='replace') as f:
			rows = f.readlines()
		print(filename)
		print('len :', len(rows))
		for row in tqdm(rows):
			count +=1
			datum = ast.literal_eval(row)
			# row = row.replace("\"", "@@@")
			# row = row.replace("\'", "\"")
			# row = row.replace("@@@", "\'")
			# try:
			# 	datum = json.loads(row.encode("utf-8","replace"))
			# except (json.decoder.JSONDecodeError, UnicodeDecodeError):
			# 	error += 1
			# 	# print(row)
			# 	continue

			# check if there's any empty required fields
			if ('gender' in datum)  and (datum['gender'] == 'M' or datum['gender'] == 'F'):
				if ('birth_year' in datum) and (isinstance(datum['birth_year'], int)) and ((2020 - int(datum['birth_year']) >= 45) or (2020 - int(datum['birth_year']) <= 35) ):
					# g = '-'
					# a = '-'
					# if 'gender' in datum:
					# 	gender+=1
					# 	g = datum['gender']
					# if 'birth_year' in datum:
					# 	age+=1
						# a = datum['birth_year']
					# if 'country' in datum:
					# 	loc+=1 
						# l = datum['country']
					# there might be more than one review per user
					for review in datum['reviews']:
						# only process if it is English
						text = ' '.join(review['text'])
						if text.strip() != '':
							if (is_english(text, lang_identifier)):
								# print(row)
								valid += 1
								# output format
								# locations, genders, ages, texts, ratings
								if filename.split('.')[0] == 'united_states':
									loc_detail = 0
								else:
									loc_detail = 1

								if datum['gender'] == 'F':
									gender_detail = 0
								else:
									gender_detail = 1

								if 2020 - int(datum['birth_year']) <= 45:
									age_detail = 0
								else:
									age_detail = 1
								rating_detail = int(review['rating'])

								# output format
								# locations, genders, ages, texts, ratings
								datum_detail = [loc_detail, gender_detail, age_detail, text, rating_detail]
							
								
								output_data.append(datum_detail)
				# if count >= 10000:
				# 	break

df_output_data = pd.DataFrame(output_data, columns=['loc', 'gender', 'age', 'text', 'rating'])

print(df_output_data.shape)
qty = 2000000
for group in ['loc', 'gender', 'age']:
	filter_is_done = False
	while not filter_is_done:
		try:
			new_df_output_data = df_output_data.groupby(group).apply(lambda x: x.sample(qty))
			df_output_data = new_df_output_data
			filter_is_done = True
			print(str(qty), ' success for ', group)
		except ValueError:
			print(str(qty), ' failed')
		qty = qty - int(0.1*qty)

# shuffle the order of the data
df_output_data = shuffle(df_output_data)

# df_output_data = df_output_data.groupby('loc').apply(lambda x: x.sample(100000))
# print(df_output_data.shape)
# df_output_data = df_output_data.groupby('gender').apply(lambda x: x.sample(50000))
# print(df_output_data.shape)
# df_output_data = df_output_data.groupby('age').apply(lambda x: x.sample(20000))
# print(df_output_data.shape)

# stratified_sample, _ = ShuffleSplit(
# 	df_output_data,
# 	# test_size=0,
# 	train_size=200,
# 	test_size=None,
# 	# stratify=df_output_data[['loc','gender', 'age', 'rating']],
# 	random_state=42,
# )

# print('test1', type(df_output_data))
# print('gender:', gender)
# print('age:', age)
# print('country:', loc)

# write the valid output data as pickle
f = open('merge.en.output_cleaned.pkl', 'wb')

cPickle.dump(df_output_data.values.tolist(), f, protocol=0)
df_output_data.to_csv('merge.en.output_cleaned.csv',encoding='utf-8')

f.close()


print('err  :', error, ' ', error/len(rows)*100)

print('valid:', valid)
print()
# break








# res = identifier.classify("nama saya dony. Di sini adalah tempat tinggal saya.")
# # res = identifier.classify("I have been living in the UK for about 10 years now.")
# print(res)




