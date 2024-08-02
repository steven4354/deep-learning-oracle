## Projects

Code implementations and notes of ML techniques

- multilayer perceptron
- cnn / convolutional layer
- autoencoder, variational autoencoder
- recurrent nn: lstm
- diffusion

## Getting started

```bash
# Activate virtual env
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Download the data for running `/projects`

```bash
cd projects/scripts
sh download.sh
```

Run the jupyter notebooks

```bash
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

## Textbooks / References

[See textbooks.md](./textbooks.md)
