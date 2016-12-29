#!/usr/bin/env bash

nvidia-docker run --rm -ti -v $(pwd):/code answeror/sigr:2016-09-21 $@
