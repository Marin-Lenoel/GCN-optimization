# A Minimal STGNN Example with MTGNN

This small project is an educational introduction to **Spatio-Temporal Graph
Neural Networks (STGNNs)**. It uses traffic data from **METR-LA** or
**PEMS-BAY** and a simplified version of **MTGNN**.

The project is intentionally small. There is no command-line interface, no
configuration system, no experiment tracker, and no plotting pipeline. The
complete experiment runs with:

```bash
python3 main.py
```

> This is a simplified, didactic model inspired by MTGNN. It is not an exact
> reproduction of the original paper and should not be used to reproduce
> published benchmark results.

## What you will learn

This example shows how to:

1. load a traffic time series and a graph adjacency matrix;
2. split time-series data into training, validation, and test sets;
3. create input windows and forecasting horizons;
4. combine temporal convolutions with graph propagation;
5. train the model and predict future traffic speeds;
6. recover predictions in the original data scale.

## The main idea

A normal time-series model only studies how values change over time. An STGNN
also studies how different locations are connected.

In these datasets:

- each **node** is a traffic sensor;
- each node has a sequence of traffic speed measurements;
- the **adjacency matrix** describes connections between sensors;
- the model uses both previous measurements and graph connections to predict
  future measurements.

The complete flow is:

```text
H5 traffic series + adjacency matrix
                 |
                 v
       chronological data split
                 |
                 v
       input and target windows
                 |
                 v
     temporal layer + graph layer
                 |
                 v
        future speed predictions
```

## Project structure

```text
for-mahin/
|-- main.py
|-- requirements.txt
|-- README.md
|-- models/
|   |-- __init__.py
|   `-- MTGNN.py
|-- shared/
|   |-- __init__.py
|   `-- loaders.py
`-- data/
    |-- metr-la.h5
    |-- metr-la-adj_mx.npy
    |-- pems-bay.h5
    `-- pems-bay-adj_mx.npy
```

- `main.py` connects all steps in one short program.
- `shared/loaders.py` loads, splits, normalizes, and creates time windows.
- `models/MTGNN.py` defines and trains the neural network.
- `data/` contains the time series and graph for both datasets.

## Datasets

| Dataset | Time steps in the full file | Sensors | Time-series shape | Adjacency shape |
|---|---:|---:|---|---|
| METR-LA | 34,272 | 207 | `(34272, 207)` | `(207, 207)` |
| PEMS-BAY | 52,116 | 325 | `(52116, 325)` | `(325, 325)` |

The H5 files store the time series. The NPY files store the adjacency matrices.
The loader adds a feature dimension, so the model receives data with shape:

```text
(time steps, sensors, 1 feature)
```

The internal H5 group names are different: METR-LA uses `df`, while PEMS-BAY
uses `speed`. The loader handles this detail automatically.

## Installation

Python 3.10 or newer is recommended.

From the `for-mahin` directory, create a virtual environment and install the
three dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The model automatically uses CUDA when PyTorch detects a compatible GPU.
Otherwise, it runs on the CPU.

## Running the example

```bash
python3 main.py
```

The default dataset is PEMS-BAY. To use METR-LA, change this line near the top
of `main.py`:

```python
DATASET_NAME = "metr-la"
```

The important settings are intentionally visible in the same file:

```python
SEQ_LEN = 12
HORIZON = 12
BATCH_SIZE = 64
EPOCHS = 2
MAX_TIMESTEPS = 2000
```

`MAX_TIMESTEPS = 2000` makes the first run reasonably fast on a CPU. The full
files are still included. To use the complete selected dataset, change it to:

```python
MAX_TIMESTEPS = None
```

Training on the complete dataset can be much slower, especially on a CPU.

## How the data loader works

### 1. Load the files

For a dataset name such as `pems-bay`, the loader reads:

```text
data/pems-bay.h5
data/pems-bay-adj_mx.npy
```

It also checks that the number of sensors in the time series matches the size
of the adjacency matrix.

### 2. Split in chronological order

The time series is divided into:

- 70% training;
- 10% validation;
- 20% test.

The series is not randomly rearranged before this split. This is important:
future observations must not be used to train a model that predicts the past.

### 3. Normalize using training data

The loader calculates the mean and standard deviation only from the training
split:

```text
normalized value = (value - training mean) / training standard deviation
```

The same training statistics are then applied to validation and test data.
This prevents information from the future from leaking into training.

### 4. Create forecasting windows

With `SEQ_LEN = 12` and `HORIZON = 12`, one example contains:

```text
X = 12 observed time steps
Y = the following 12 time steps
```

The tensor shapes are:

```text
X: (batch, 12, number of sensors, 1)
Y: (batch, 12, number of sensors, 1)
```

The `TrafficDataset` creates these windows only when the DataLoader requests
them. It does not copy every possible window into memory in advance.

## How the simplified MTGNN works

The model contains five main ideas.

### 1. Input projection

A `1 x 1` convolution transforms the single input feature into hidden channels.

### 2. Temporal gated convolution

The temporal layer looks for patterns across recent time steps. A filter learns
candidate information, while a gate learns how much of that information should
pass to the next layer.

### 3. Graph propagation

`MixProp` sends information between connected sensors. Repeating this operation
allows a sensor to receive information from nodes that are multiple graph hops
away.

### 4. Adaptive graph

The model learns node embeddings and uses them to build an additional adjacency
matrix. The simplified implementation combines:

```text
50% provided adjacency + 50% learned adjacency
```

This lets the model use known road connections while also learning other useful
relationships from the data.

### 5. Output projection

The final layers convert the hidden representation into one prediction for
every sensor and every future time step.

## Training and output

The model trains with mean squared error (MSE). Validation loss is used to keep
the best model state in memory.

After training, `main.py` prints:

- training and validation loss for each epoch;
- normalized test MSE;
- test MAE in the original speed scale;
- the shapes of true and predicted values;
- one example for sensor 0.

The `main` function also returns the complete arrays:

```python
from main import main

y_true, y_pred = main()
```

Both arrays have this shape:

```text
(test examples, forecast horizon, sensors, features)
```

For the default educational run with PEMS-BAY, this is:

```text
(378, 12, 325, 1)
```

The returned values are already converted back to the original data scale.

## Suggested reading order

Read the project in this order:

1. `main.py` to see the complete experiment;
2. `shared/loaders.py` to understand the data shapes and windows;
3. `models/MTGNN.py`, starting with `forward`, to follow the model;
4. the `fit`, `evaluate`, and `predict` methods to understand training.

When reading the code, keep track of these four dimensions:

```text
B = batch size
T = time steps
N = graph nodes or sensors
C = feature or hidden channels
```

The DataLoader uses `(B, T, N, C)`. Inside the model, the tensor is changed to
`(B, C, N, T)` because this order is convenient for PyTorch convolutions.

## Important limitations

- This is an educational MTGNN-like model, not the complete original MTGNN.
- The default run uses only 2,000 time steps and two epochs.
- The code uses one global mean and standard deviation for normalization.
- It does not use time-of-day features, day-of-week features, or missing-value
  masks.
- It does not perform hyperparameter search or repeated runs with many seeds.
- The printed metrics are useful for checking the learning pipeline, but they
  are not directly comparable with published benchmark results.

## Small experiments to try

After the first successful run, try one change at a time:

1. switch from PEMS-BAY to METR-LA;
2. increase `EPOCHS` and observe the validation loss;
3. set `MAX_TIMESTEPS = None` if enough compute is available;
4. change `HORIZON` and inspect the output shape;
5. temporarily replace the adjacency matrix with an identity matrix and study
   what happens when each sensor only sees itself;
6. compare predictions from an untrained model with predictions after training.

These experiments help separate the temporal, spatial, and optimization parts
of an STGNN.
