## Projects

Code implementations and notes of ML techniques

- multilayer perceptron
- cnn / convolutional layer
- autoencoder, variational autoencoder
- recurrent nn: lstm
- diffusion

## Getting started


```bash
poetry install
poetry run jupyter notebook
```

Download the data for running `/projects`

```bash
cd projects/scripts
sh download.sh
```

Run the jupyter notebooks

```bash
poetry run jupyter notebook
```

Make sure to save new deps when installing new ones

```bash
poetry add <package-name>
```

Adding large files to git lfs

```bash
# Replace json with any other file types that are large (.csv, h5 etc)
git lfs track "*.json"
```

## Textbooks / References

[See textbooks.md](./textbooks.md)
