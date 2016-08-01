:
sudo --set-home docker run \
  --rm \
  --name deepdream-run \
  -e MKL_NUM_THREADS=$MKL_NUM_THREADS \
  -e MKL_DOMAIN_NUM_THREADS="$MKL_DOMAIN_NUM_THREADS" \
  -e MKL_DYNAMIC=$MKL_DYNAMIC \
  -v "$PWD":/data \
  rhee/deepdream /data/run.sh "$@"
