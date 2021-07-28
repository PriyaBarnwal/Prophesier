# Image Quality Assessment
  Original Repo = https://github.com/idealo/image-quality-assessment

## Getting started

1. Install [jq](https://stedolan.github.io/jq/download/)

2. Install [Docker](https://docs.docker.com/install/)

3. Build docker image `docker build -t nima-cpu . -f Dockerfile.cpu`


## Predict
- Create a folder called test_images within src directory to store the images
- Input CSV = src/dataset.csv
- Output CSV = src/output/output.csv
- Run the predict script:
    ```bash
    ./predict  \
    --docker-image nima-cpu \
    --base-model-name MobileNet
    ```
