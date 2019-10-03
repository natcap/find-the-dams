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

# build the python environments for 3.6 and 3.7 using conda, pip, and
# python setup.py install
SHELL ["/bin/bash", "-c"]

RUN conda create -y --name py36 python=3.6
RUN conda run -v -n py36 conda install -c conda-forge gdal=2.4.2 && conda clean -a -y
RUN conda run -v -n py36 pip install --no-cache-dir \
    cython \
    ecoshard \
    flask \
    pytest \
    pytest-cov \
    mock \
    numpy \
    pillow \
    requests \
    retrying \
    rtree \
    scipy \
    setuptools-scm \
    shapely \
    sympy \
    taskgraph && conda clean -a -y

RUN conda init bash && echo "source activate py36" > ~/.bashrc

RUN hg clone http://bitbucket.org/natcap/pygeoprocessing /usr/local/pygeoprocessing
WORKDIR /usr/local/pygeoprocessing
RUN hg update 1.8.0 && conda run -v -n py36 python setup.py install
WORKDIR /workspace
COPY tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl /workspace
RUN conda run -v -n py36 pip install tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl && rm tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl

ENTRYPOINT ["/bin/bash", "-l", "-c"]
