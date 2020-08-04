import os
import _pickle as cPickle
# import json
import ast
from tqdm import tqdm
from langid.langid import LanguageIdentifier, model
from sklearn.model_selection import train_test_split
import pandas as pd

a = [[1,2,3,4,0], [5,6,7,8,9]]

df_a = pd.DataFrame(a, columns=['a','b','x','y','z'])

print(a)
print(a == df_a.values.tolist())

f = open('buangggg', 'wb')
cPickle.dump(df_a.values.tolist(), f, protocol=0)