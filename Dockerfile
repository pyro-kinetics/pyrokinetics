from python:3.9

RUN apt-get update
RUN apt-get -y install gfortran

COPY ./ /pyrokinetics
RUN cd /pyrokinetics && pip install .

