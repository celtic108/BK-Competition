import csv
import numpy as np

ifile = open('../Data/train_numeric.csv', 'rb')
numeric = csv.reader(ifile)

header = numeric.next()
data = []
for row in numeric:
    data.append(row)
data = np.array(data)
print header
ifile.close()
