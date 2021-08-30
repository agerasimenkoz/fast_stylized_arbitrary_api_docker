# Fast arbitrary image style transfer for Fast API in docker

## How to run
* Install and run Docker
* Build Docker image using `docker build . -t styled_server`
* Run Docker container using `docker run --rm -it -p 80:80 styled_server`
* Go to `http://127.0.0.1:80/docs` to see all available methods of the API

**All parameters are hardcoded to make the example as easy as possible** 
## Used libs
* [Fast API](https://fastapi.tiangolo.com/)
* [Docker](https://www.docker.com/)
* [tensorflow](https://www.tensorflow.org)

## Source code
* [server.py](server.py) contains API logic
* [train.py](train.py) trains dummy model using Iris dataset
* [query_example.py](query_example.py) helps to check that docker container working properly
* [Dockerfile](Dockerfile) describes a Docker image that is used to run the API
