import urllib.request

import pandas as pd

# Script to download images
dataset = pd.read_csv('/Users/prashant.gupta/Downloads/EP/ABC_VPC_Nov19_1.csv', quotechar='"', skipinitialspace=True)
rows = dataset.shape[0]

for row in range(rows):
    print('downloading row ', row, '\r')
    url = dataset.image_url[row]
    filename = url
    try:
        urllib.request.urlretrieve('http://' + url, '/Users/prashant.gupta/Downloads/EP/images/' + filename[filename.rfind('/') + 1: len(filename)] + '.jpg')
    except:
        print('Error downloading image(s) at row ', row)
        with open('log.txt', 'a') as log:
            log.write('Error row ' + str(row))
