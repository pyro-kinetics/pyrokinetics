from python:3.11

RUN apt-get update
RUN apt-get -y install gfortran && apt-get clean

COPY . /pyrokinetics
WORKDIR /pyrokinetics

ARG SETUPTOOLS_SCM_PRETEND_VERSION
RUN pip install --no-cache-dir .[tests]

CMD [ "ipython" ]

