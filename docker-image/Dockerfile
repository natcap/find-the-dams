FROM conda/miniconda3

# install build essentials & mercurial then delete the cache to minimize
# the docker size:
#   build-essential to build cython extensions
#   libspatialindex-c4v5 for the rtree python extension
#   mercurial for source control repo of PyGeoprocessing
USER root
RUN apt-get update \
&& apt-get install -y \
    build-essential \
    libspatialindex-c4v5 \
    mercurial \
&& rm -rf /var/lib/apt/lists/*

COPY install-pip-packages.sh .

# build the python environments for 3.6 and 3.7 using conda, pip, and
# python setup.py install
RUN /bin/bash -c "\
    conda create -y --name py37 python=3.7 \
    && conda clean -a -y"
RUN /bin/bash -c "conda run -v -n py37 conda install -c conda-forge gdal=2.4.1"
RUN /bin/bash ./install-pip-packages.sh 3 7

RUN hg clone http://bitbucket.org/natcap/pygeoprocessing
WORKDIR pygeoprocessing
RUN hg update develop
RUN /bin/bash -c "conda run -v -n py37 python3 setup.py install"
RUN /bin/bash -c "conda run -v -n py37 pip install tensorflow retrying requests taskgraph flask pillow"
