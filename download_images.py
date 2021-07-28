import urllib.request

import pandas as pd

# Script to download images from dataset.csv

dataset = pd.read_csv('dataset_28July2021.csv', quotechar='"', skipinitialspace=True)
rows = dataset.shape[0]

for row in range(rows):
    print('downloading row ', row, '\r')
    url = dataset.image[row]
    filename = url[url.rfind('/') + 1:url.rfind('?')]

    try:
        urllib.request.urlretrieve(url, 'images/' + filename)
    except:
        print('Error downloading image(s) at row ', row)
        with open('log.txt', 'a') as log:
            log.write('Error row ' + str(row))
