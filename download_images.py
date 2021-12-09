import urllib.request

import pandas as pd

# Script to download images
dataset = pd.read_csv('/Users/prashant.gupta/Downloads/EP/ABC_VPC_Nov19_1.csv', quotechar='"', skipinitialspace=True)
rows = dataset.shape[0]

for row in range(rows):
    print('downloading row ', row, '\r')
    try:
        if type(dataset.image_url[row]) != str: continue
        url = dataset.image_url[row]
        # filename = url[url.rfind('/') + 1 : len(url)] + '.jpg' if url.startswith('s3') else url[url.rfind('/') + 1 : url.rfind('?')]
        # url = 'http://' + url if url.startswith('s3') else url
        urllib.request.urlretrieve('http://' + url, '/Users/prashant.gupta/Downloads/EP/images1/' + url[url.rfind('/') + 1 : len(url)] + '.jpg') if url.startswith('s3') else print('Not downloading - ' + url)
    except:
        print('Error downloading image(s) at row ', row)
        with open('log.txt', 'a') as log:
            log.write('Error row ' + str(row))
