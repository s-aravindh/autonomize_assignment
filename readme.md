
# To run the app.

## install requirements
```
pip install -r src/requirements.txt
```

## start the server:
```
cd src
uvicorn run app:app
```


## sample request:
```curl
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_type": "fixed",
  "dna_char": "GGGNCGCCCNCTCTTAGGGGGAANNCATTTNGACTGNGTNCGTNTGCAAATACTGNANNTGCCGTGTAATTATNNCGNTACTGTTNNGCNCCACNGCCCAGNAGNTGAGNG"
}'
```

model_type is `fixed` or `padded`. if dna_char is less than 128. it should be padded