#!/bin/sh -x

# NOTICE: never directly source compilervars.sh in a shell script
# compilervars.sh inappropriately modifies $n variables
fix_env(){(
    . /opt/intel/bin/compilervars.sh intel64
    env | sed -e 's/^/export /'
)}
eval $(fix_env)

exec python -u /deepdream.py "$@"
