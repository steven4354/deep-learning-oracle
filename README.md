## Projects

Code implementations of ML techniques (multilayer perceptron, cnn, autoencoders, lstm, transformers)

## Getting started

```bash
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

- `projects/data/full_format_recipes.json` is from this kaggle data set: https://www.kaggle.com/datasets/hugodarwood/epirecipes

## Textbooks / References

Doing the exercises from "Deep Learning with Pytorch". [Amazon link](https://www.amazon.com/Deep-Learning-PyTorch-Eli-Stevens/dp/1617295264)

Added "Generative Deep Learning". [Amazon link](https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1098134184)

### Topics covered

#### Deep learning with pytorch (textbook)

```
long short-term memory (LSTM) modules: no, says out of scope of the book to cover

recurrent neural networks: yes

transformer networks: no, says out of scope of the book to cover

attention: no

CNN: yes

encoder/decoder: no

Sequence Transduction Models: ??
```

#### Generative deep learning (textbook)

```
transformer networks: yes

long short-term memory (LSTM) model: yes, one or two exercises to train one from scratch

attention: yes (basically the transformer chapter)

encoder/decoder: yes

Sequence Transduction Models: ??
```
