import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from functools import partial
import sqlite3
import pandas as pd
import sqlalchemy as sqla

by_letter = defaultdict(list)
words = ['apple', 'bat', 'bar', 'atom', 'book']
for word in words:
    by_letter[word[0]].append(word)
print(by_letter)
a = {1,2,3,4,5}
b = {3,4,5,6,7,8}
print(a.union(b))
print(a|b)
print(a.intersection(b))
print(a&b)
# hash([1,2,3]) # keys of a dict must be immutable objects which are hashable

strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
print([len(x) for x in strings])
print({len(x) for x in strings})
print(set(map(len, strings)))
loc_mapping = {val: index for index, val in enumerate(strings)}
print(loc_mapping)

all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'], ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]
result = [name for names in all_data for name in names if name.count('e') >= 2]
print(result)

states = ['   Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda', 'south   carolina##', 'West virginia?']
def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()  # remove leading and trailing whitespaces
        value = re.sub('[!#?]', "", value)
        value = value.title()
        result.append(value)
    return result
print(clean_strings(states))

def apply_to_list(some_list, f):
    return [f(x) for x in some_list]
ints = [4, 0, 1, 5, 6]
print(apply_to_list(ints, lambda x: x * 2))
print([x*2 for x in ints])

def add_numbers(x, y):
    return x + y
add_five = partial(add_numbers, 5)
print(add_five(1))

nsteps = 1000
nwalks = 5000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
print(steps.cumsum())
hits30 = (np.abs(walks) >= 30).any(1)
print(walks)
print(hits30)
print(len(hits30))
print(hits30.sum())
print((np.abs(walks) >= 30).any(1).sum())
print(len((np.abs(walks[hits30]) >= 30).argmax(1)))

query = """CREATE TABLE test (a VARCHAR(20), b VARCHAR(20), c REAL, d INTEGER);"""
con = sqlite3.connect('E:\\mydata.sqlite')
con.execute(query)
con.commit()
data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
con.executemany(stmt, data)
con.commit()
cursor = con.execute('select * from test')
rows = cursor.fetchall()
rows
print(pd.DataFrame(rows, columns=[x[0] for x in cursor.description]))
db = sqla.create_engine('sqlite:///E:\\mydata.sqlite')
pd.read_sql('select * from test', db)
print(pd.read_sql('select * from test', db))

