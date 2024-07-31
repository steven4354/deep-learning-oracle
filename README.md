## Projects

Code implementations of ML techniques 
- multilayer perceptron
- cnn / convolutional layer
- autoencoder, variational autoencoder
- recurrent nn: lstm

## Getting started

```bash
# Download the relevant data
# Run the jupyter notebooks
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Make sure to save new deps when installing new ones

```bash
pip freeze requirements.txt
```

Adding large files to git lfs

```bash
# Replace json with any other file types that are large (.csv, h5 etc)
git lfs track "*.json"
```

## Data

- `projects/data/full_format_recipes.json` from this kaggle data set: https://www.kaggle.com/datasets/hugodarwood/epirecipes

## Textbooks / References

[See textbooks.md](./textbooks.md)
