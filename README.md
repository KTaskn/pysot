# PySOT

See the [official repository](https://github.com/STVIR/pysot) for more information about the code contents.

You can execute pysot with docker.

``` sh
> docker build . --tag=<your tag>
> docker run --rm -ti --gpus all -v <your path>:/video <your tag> bash
$ sh execute.sh
```