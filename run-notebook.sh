:
sudo --set-home docker run --name=dream-notebook --rm -t -i -p 8888:8888 -v "$PWD":/data rhee/deepdream ${1-jupyter notebook --ip=0.0.0.0}
