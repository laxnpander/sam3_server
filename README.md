# SAM3 Server

# Docker

```shell
# Build docker container
./.docker/build.sh

# Run docker container
./.docker/run.sh
```

# Usage

1. Upload image

```shell
curl -X 'POST' \
  'http://127.0.0.1:8000/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test.png;type=image/png'
```

2. Run Inference
```shell
curl -X 'POST' \
  'http://127.0.0.1:8000/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "filepaths": [
    "test.png"
  ],
  "prompt": "test"
}'
```