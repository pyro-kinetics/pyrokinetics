from python:3.11

RUN apt-get update
RUN apt-get -y install gfortran && apt-get clean

COPY . /pyrokinetics
WORKDIR /pyrokinetics
RUN git describe --tags > VERSION
RUN pip install --no-cache-dir .[tests]

CMD [ "ipython" ]

