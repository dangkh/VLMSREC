# VLMSREC
```
VLMSREC/
│
├── data/
│   └── <dataset_name>/
│       ├── <dataset_name>.inter
│       ├── <dataset_name>_5.json
│       ├── i_id_mapping.csv
│       └── meta_<dataset_name>.json
├── README.md
├── src
│	├──gen_image_feat.py              # Script to generate text embeddings
│	├──vlm2text.py
│   └─requirements.txt
└───main.py
```

### 2. Preprocess

#### VLM read images

Vlm generate feature for image, edit src/config.py to change the dataset and vlm model.
```sh
python src/vlm2feat.py
```
#### Generate image features 

```sh
python src/gen_image_feat.py
```

### 2. Preprocess
```sh
python src/main.py -d baby -m DRAGON
```

baby/sport ... DRAGON/FREEDOM/MENTOR