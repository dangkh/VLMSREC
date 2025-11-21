# VLMSREC

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
│	├──get_text_feat.py              # Script to generate text embeddings
│	├──README.md
│   └─requirements.txt
├───main.py
├───vlm2text.py
└───img2text.py


### 2. Preprocess

#### Generate text features for images

Vlm generate feature for image, edit src/config.py to change the dataset and vlm model.
```sh
python src/vlm2feat.py
```