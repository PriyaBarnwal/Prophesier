from urllib.request import urlopen

import numpy as np
import scipy.misc
import uvicorn as uvicorn
from dateutil.parser import parse
from fastapi import FastAPI
from keras.models import load_model
from pydantic import BaseModel
from scipy.misc import imread

IMAGE_SIZE = 256
CHANNELS = 3
model = load_model('my_model_full.h5')

app = FastAPI()

def read_hashtags():
    with open('hashtags.txt', 'r') as file:
        hashtags = file.read().split()
        hashtags = dict(enumerate(hashtags))
        hashtags = {v: k for k, v in hashtags.items()}
    return hashtags

hashtags = read_hashtags()

def finalFeatureSet(data, item):
    dataMatrix = []
    imageMatrix = []
    for iUrl in item.images:
        with urlopen(iUrl) as file:
            img = imread(file, mode='RGB')
            img = scipy.misc.imresize(img, (IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
            img = img / 255
            imageMatrix.append(img)
            dataMatrix.append(data)

    imageMatrix = np.asarray(imageMatrix)
    dataMatrix = np.asarray(dataMatrix)

    return [imageMatrix, dataMatrix]


def metaDataFeatures(item):
    date = parse(item.publishDate)
    mydate = date.year * 365 + date.month * 30 + date.day
    tags = [i for i in item.caption.split() if i.startswith("#")]
    tags = [tag[1:] for tag in tags]
    tagsNumber = len(tags)
    tagsValues = [10000 - hashtags[tag] for tag in tags if tag in hashtags]
    tagsSum = sum(tagsValues)
    data = [item.nfollowers, mydate, date.weekday(), tagsNumber, tagsSum]
    return data


class Input(BaseModel):
    caption = ""
    publishDate = ""
    nfollowers = 0
    images = []

@app.get("/ping")
async def ping():
    return "yo"


@app.post("/predict")
async def predict(item: Input):
    data = metaDataFeatures(item)
    features = finalFeatureSet(data, item)
    prediction = model.predict(features)
    return prediction.tolist()



if __name__ == "__main__":
    uvicorn.run("app:app", debug=False, host='0.0.0.0', port=8895, workers=3)
