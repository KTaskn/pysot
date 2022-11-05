FROM ktaskn/jupyter-pytorch

RUN python -m pip install -U pip \
    && python -m pip install pyyaml yacs tqdm colorama cython tensorboardX

COPY ./ /work

WORKDIR /work

RUN python setup.py build_ext --inplace

