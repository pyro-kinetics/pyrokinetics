from python:3.9


COPY ./ /pyrokinetics
RUN cd /pyrokinetics && pip install .

ENTRYPOINT python3
