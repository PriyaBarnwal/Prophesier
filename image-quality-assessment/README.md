# Image Quality Assessment
  Original Repo = https://github.com/idealo/image-quality-assessment
  
## Getting started

1. Install [jq](https://stedolan.github.io/jq/download/)

2. Install [Docker](https://docs.docker.com/install/)

3. Build docker image `docker build -t nima-cpu . -f Dockerfile.cpu`


## Predict
    ```bash
    ./predict  \
    --docker-image nima-cpu \
    --base-model-name MobileNet
    ```
