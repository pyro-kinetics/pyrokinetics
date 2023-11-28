from python:3.9

RUN apt-get update
RUN apt-get -y install gfortran && apt-get clean

COPY ./ /pyrokinetics
RUN cd /pyrokinetics && git describe --tags > VERSION
RUN cd /pyrokinetics && pip install --no-cache-dir . && rm -rf .git

CMD [ "python" ]

