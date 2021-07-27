
import os
import glob
import json
import argparse
import urllib.request
from utils.utils import calc_mean_score, save_json
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator
import csv


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)


def main(base_model_name, csv_file, img_format='jpg'):
    print(base_model_name)
    #print(weights_file)
    print(csv_file)
    # load samples
    # if os.path.isfile(image_source):
    #     image_dir, samples = image_file_to_json(image_source)
    # else:
    #     image_dir = image_source
    #     samples = image_dir_to_json(image_dir, img_type='jpg')

    # build model and load weights
    nima1 = Nima(base_model_name, weights=None)
    nima1.build()
    nima1.nima_model.load_weights('/src/weights_technical.hdf5')

    nima2 = Nima(base_model_name, weights=None)
    nima2.build()
    nima2.nima_model.load_weights('/src/weights_aesthetic.hdf5')


    # initialize data generator
    # data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(),
    #                                    img_format=img_format)

    # # get predictions
    # predictions = predict(nima.nima_model, data_generator)

    # scores = []
    # # calc mean scores and add to samples
    # for i, sample in enumerate(samples):
    #     sample['mean_score_prediction'] = calc_mean_score(predictions[i])
    #     scores.append(sample['mean_score_prediction'])

    # print(json.dumps(samples, indent=2))


    with open(csv_file, 'r') as read_obj, \
        open('/src/output/output.csv', 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = csv.reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = csv.writer(write_obj)
        headers = next(csv_reader)
        headers.append('technical_score')
        headers.append('aesthetic_score')
        csv_writer.writerow(headers)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Append the default text in the row / list
            try:
                if(row[-1]):
                    imageName= row[-1].split('?')[0].split('/')[-1]
                    urllib.request.urlretrieve(row[-1], f'/src/test_images/{imageName}')
                    image_dir, samples = image_file_to_json(f'/src/test_images/{imageName}')
                    data_generator1 = TestDataGenerator(samples, image_dir, 64, 10, nima1.preprocessing_function(),
                                        img_format=img_format)
                    data_generator2 = TestDataGenerator(samples, image_dir, 64, 10, nima2.preprocessing_function(),
                                        img_format=img_format)
                    predictions1 = predict(nima1.nima_model, data_generator1)
                    predictions2 = predict(nima2.nima_model, data_generator2)
                    row.append(calc_mean_score(predictions1[0]))
                    row.append(calc_mean_score(predictions2[0]))
                    # Add the updated row / list to the output file
                    csv_writer.writerow(row)
            except:
                print('Error downloading image ', row[1])
        
    # if predictions_file is not None:
    #     save_json(samples, predictions_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-csv', '--csv-file', help='file with predictions', required=True)
    args = parser.parse_args()

    main(**args.__dict__)
