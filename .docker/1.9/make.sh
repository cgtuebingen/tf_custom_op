sudo docker build -t tensorflow:cuda9-ubuntu16.04-depend -f Dockerfile.depend-cuda9-ubuntu16.04 .
sudo docker build -t tensorflow:cuda9-ubuntu16.04-tf1.9 -f Dockerfile-tf1.9-cuda9-ubuntu16.04 .

