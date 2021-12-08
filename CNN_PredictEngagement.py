import math
from os import listdir
from os.path import isfile, join
import keras
import numpy as np
import pandas as pd
import scipy.misc
import re
from dateutil.parser import parse
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Model, load_model

def extract_hashtags(text):

    # the regular expression
    regex = "#(\w+)"

    # extracting the hashtags
    hashtag_list = re.findall(regex, text)

    # printing the hashtag_list
    # print("The hashtags in \"" + text + "\" are :")
    # for hashtag in hashtag_list:
    #     print(hashtag)
    return hashtag_list

def read_hashtags():
    with open('hashtags.txt', 'r') as file:
        hashtags = file.read().split()
        hashtags = dict(enumerate(hashtags))
        hashtags = {v:k for k,v in hashtags.items()}
        return hashtags

COLUMNS = 5

print('Loading hashtags and dataset ...')
dataset = pd.read_csv('/Users/prashant.gupta/Downloads/EP/ABC_VPC_Nov19-Table 1.csv', quotechar='"', skipinitialspace=True)
# dataset.dropna()
dataset = dataset.loc[dataset['instagram_post_likes'] > 50]
dataset = dataset.head(100)
hashtags = read_hashtags()

print('Loading image filenames...')
dir = '/Users/prashant.gupta/Downloads/EP/images/'
filenames = [f for f in listdir(dir) if isfile(join(dir, f))]

print('Building matrix...')
matrix = []
matrixFilenames = []

for index, row in dataset.iterrows():
    # Build filename from url
    # filename = row.image[row.image.rfind('/') + 1:row.image.rfind('?')]
    filename = row.image_url[row.image_url.rfind('/') + 1 : len(row.image_url)] + '.jpg'
    if not filename in filenames: continue

    matrixFilenames.append(filename)
    # nposts = row.numberPosts
    # nfollowing = row.numberFollowing
    nfollowers = 0.0 if math.isnan(row.followers) else row.followers
    numberLikes = row.instagram_post_likes
    # technical_score = row.technical_score
    # aesthetic_score = row.aesthetic_score
    date = parse(row.created_time)
    mydate = date.year * 365 + date.month * 30 + date.day
    # mentions = len(ast.literal_eval(row.mentions))
    # meanNumberLikes = m.loc[m.index == row.alias].numberLikes[0]

    # hashtag analysis
    # tags = row.tags.split(" ")
    tags = extract_hashtags(row.text)
    tags = [tag[1:] for tag in tags]
    tagsNumber = len(tags)
    tagsValues = [10000 - hashtags[tag] for tag in tags if tag in hashtags]
    tagsSum = sum(tagsValues)
    # likes_ratio = numberLikes/nfollowers

    # data = [nfollowers, mydate, date.weekday(), tagsNumber, technical_score, aesthetic_score, numberLikes]
    data = [mydate, date.weekday(), tagsNumber, tagsSum, numberLikes]
    matrix.append(data)

print('Building model...')
TRAIN = int(0.9*len(dataset))
matrix = np.array(matrix)
X = matrix[:, :COLUMNS - 1]
y = matrix[:, COLUMNS - 1]
X_train, X_test = np.split(X, [TRAIN])
y_train, y_test = np.split(y, [TRAIN])

IMAGE_SIZE = 256
CHANNELS = 3

# Convolutional neural network for images
image_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name='image_input')
image_nn = Conv2D(4, (5, 5), activation='relu')(image_input)
image_nn = Conv2D(16, (3, 3), activation='relu')(image_input)
image_nn = MaxPooling2D(pool_size=(2, 2))(image_nn)
image_nn = Conv2D(32, (3, 3), activation='relu')(image_nn)
image_nn = Conv2D(64, (3, 3), activation='relu')(image_nn)
image_nn = MaxPooling2D(pool_size=(2, 2))(image_nn)
image_nn = Conv2D(128, (3, 3), activation='relu')(image_nn)
image_nn = MaxPooling2D(pool_size=(2, 2))(image_nn)
image_nn = Dropout(0.2)(image_nn)
image_nn = Conv2D(256, (3, 3), activation='relu')(image_nn)
image_nn = MaxPooling2D(pool_size=(2, 2))(image_nn)
image_nn = Dropout(0.2)(image_nn)
image_nn = Flatten()(image_nn)

# Neural network for user data and image metadata
data_input = Input(shape=(COLUMNS - 1,), name='data_input')
data_nn = Dense(COLUMNS, activation='relu')(data_input)

# Merge of the networks and last part of the network
output_nn = keras.layers.concatenate([image_nn, data_nn])
output_nn = Dense(500, activation='relu')(output_nn)
output_nn = Dropout(0.1)(output_nn)
output_nn = Dense(10, activation='relu')(output_nn)
output_nn = Dense(1, name='output')(output_nn)

# Model definition
model = Model(inputs=[image_input, data_input], outputs=[output_nn])

# Compile model
model.compile(loss='mape', optimizer='adam', metrics=['mape', 'mae', 'mse'])

def seqGenerator(X_train, y_train, batch_size, dir, filenames):
    index = 0
    num_train = X_train.shape[0]
    while 1:
        images_batch = []
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            img = scipy.misc.imread(join(dir, filenames[index]), mode='RGB')
            img = scipy.misc.imresize(img, (IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
            img = img / 255
            images_batch.append(img)
            x_batch.append(X_train[index])
            y_batch.append(y_train[index])
            index = index + 1 if not index == num_train - 1 else 0

        images_batch = np.asarray(images_batch)
        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)
        yield [images_batch, x_batch], y_batch




def seqTestGenerator(X_test, y_test, batch_size, dir, filenames):
    index = 0
    num_train = X_test.shape[0]
    print("lopping ",index)
    images_batch = []
    x_batch = []
    y_batch = []
    for i in range(batch_size):
        img = scipy.misc.imread(join(dir, filenames[len(X_train)]), mode='RGB')
        img = scipy.misc.imresize(img, (IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        img = img / 255
        images_batch.append(img)
        x_batch.append(X_test[index])
        y_batch.append(y_test[index])
        index = index + 1 if not index == num_train-1 else 0

    images_batch = np.asarray(images_batch)
    x_batch = np.asarray(x_batch)
    y_batch = np.asarray(y_batch)
    yield [images_batch, x_batch]



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


print('Training...')
# Fit the model
trainingset_size = len(X_train)
batch_size = 25
losshistory = LossHistory()
model.fit_generator(seqGenerator(X_train, y_train, batch_size, dir, matrixFilenames),
                     steps_per_epoch=30, epochs=20, callbacks=[losshistory])


# save the model to disk
model.save('ABC_Nov19.h5')
# model = load_model('gopro_5_feature.h5')
#
# print("model loaded")

print("Evaluate model on test data")
score = model.evaluate(seqTestGenerator(X_test, y_test, 5, dir, matrixFilenames), verbose=0)
print("test loss, test acc:", score)

# Generate a prediction using model.predict()
# and calculate it's shape:
print("Generate a prediction")
prediction = model.predict(seqTestGenerator(X_test, y_test, 5, dir, matrixFilenames))
print("prediction shape:", prediction.shape)
print("actual ---> prediction")
print(y_test[:10], "--->", prediction.flatten())
