# Multilayer Perceptron

> _*Summary: This project is an introduction to artificial neural networks, with the implementation of a multilayer perceptron*_

| Requirements                                                       | Skills                                |
| ------------------------------------------------------------------ | ------------------------------------- |
| - `python3.11`<br> - `numpy`<br> - `pandas`<br> - `matplotlib`<br> | - `DB & Data`<br> - `Algorithms & AI` |

</br>

## Project Overview

This project is based on the multilayer perceptron from 42 Paris, and I have added more functionalities. For those who want to refer to this project within the scope of 42, here is the link to the [multilayer perceptron - 42 Project](https://github.com/jmcheon/multilayer_perceptron/tree/42v2.0).

</br>

## 1. Dataset

It is a CSV file of 32 columns, the column **diagnosis** being the label you want to learn given all the other features of an example, it can be either the value $M$ or $B$ (for malignant or benign).

The features of the dataset describe the characteristics of a cell nucleus of breast mass extracted with [fine-needle aspiration](https://en.wikipedia.org/wiki/Fine-needle_aspiration). (for more detailed information, go [here](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names)).

#### 5 examples of the dataset

```bash
842302,M,17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
842517,M,20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902
84300903,M,19.69,21.25,130,1203,0.1096,0.1599,0.1974,0.1279,0.2069,0.05999,0.7456,0.7869,4.585,94.03,0.00615,0.04006,0.03832,0.02058,0.0225,0.004571,23.57,25.53,152.5,1709,0.1444,0.4245,0.4504,0.243,0.3613,0.08758
84348301,M,11.42,20.38,77.58,386.1,0.1425,0.2839,0.2414,0.1052,0.2597,0.09744,0.4956,1.156,3.445,27.23,0.00911,0.07458,0.05661,0.01867,0.05963,0.009208,14.91,26.5,98.87,567.7,0.2098,0.8663,0.6869,0.2575,0.6638,0.173
84358402,M,20.29,14.34,135.1,1297,0.1003,0.1328,0.198,0.1043,0.1809,0.05883,0.7572,0.7813,5.438,94.44,0.01149,0.02461,0.05688,0.01885,0.01756,0.005115,22.54,16.67,152.2,1575,0.1374,0.205,0.4,0.1625,0.2364,0.07678
...
```

#### Attribute Information

1. ID number
2. Diagnosis (M = malignant, B = benign)  
   3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from the center to points on the perimeter)  
b) texture (standard deviation of gray-scale values)  
c) perimeter  
d) area  
e) smoothness (local variation in radius lengths)  
f) compactness (perimeter^2 / area - 1.0)  
g) concavity (severity of concave portions of the contour)  
h) concave points (number of concave portions of the contour)  
i) symmetry  
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error, and "worst" or largest (mean of the three  
largest values) of these features were computed for each image,  
resulting in 30 features. For instance, field 3 is Mean Radius, field  
13 is Radius SE, field 23 is Worst Radius.

All feature values are recorded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

</br>

## 2. Implementation

#### Usage

```bash
usage: main.py [-h] [-s SPLIT] [-t] [-p] [-c [{models,optimizers}]]

This program is an implementation of a Multilayer Perceptron (MLP), a type of artificial neural network designed for tasks such as classification, regression, and pattern recognition.

options:
  -h, --help            show this help message and exit
  -s SPLIT, --split SPLIT
                        Split dataset into train and validation sets.
  -t, --train           Train with dataset.
  -p, --predict         Predict using saved model.
  -c [{models,optimizers}], --compare [{models,optimizers}]
                        Compare models by plotting learning curves.
```

For training with the saved model topology

```
python3 main.py --topology topologies/binary_sigmoid.json -t
```

It saves its weight and bias in a `binary_sigmoid.npz` file

<br>

For prediction

```
python3 main.py --topology topologies/binary_sigmoid.json -p
```

### Binary Classification

#### Neural Network Topology

```python
input_shape = 30
output_shape = 1
loss='binary_crossentropy', learning_rate=1e-3, batch_size=1, epochs=30

network = model.create_network([
        Dense(input_shape, 20, activation='relu'),
        Dense(20, 10, activation='relu'),
        Dense(10, 5, activation='relu'),
        Dense(5, output_shape, activation='sigmoid')
        ])
```

#### Training Learning Curves

| loss, accuracy for training and validation                                                                                |
| ------------------------------------------------------------------------------------------------------------------------- |
| ![learning_curves](https://github.com/jmcheon/multilayer_perceptron/assets/40683323/36faa30e-57ea-4043-80f6-b66ad092fe1e) |

Train accuracy: 0.9845 Validation accuracy: 0.9912

---

### Comparing models - 3 models of different neural networks

```python
input_shape = 30
output_shape = 1
loss='binary_crossentropy', learning_rate=1e-3, batch_size=1, epochs=50

    model1 = Model()
    model1.create_network([
        Dense(input_shape, 20, activation='relu'),
        Dense(20, 10, activation='relu'),
        Dense(10, 5, activation='relu'),
        Dense(5, output_shape, activation='sigmoid')
        ])

    model2 = Model()
    model2.create_network([
        Dense(input_shape, 15, activation='relu'),
        Dense(15, 5, activation='relu'),
        Dense(5, output_shape, activation='sigmoid')
        ])

    model3 = Model()
    model3.create_network([
        Dense(input_shape, 5, activation='relu'),
        Dense(5, output_shape, activation='sigmoid')
        ])
```

| models                                                                                                           |
| ---------------------------------------------------------------------------------------------------------------- |
| ![models](https://github.com/jmcheon/multilayer_perceptron/assets/40683323/accc3f5d-06c9-441b-89cd-1272494d9f5d) |

Model1 - Train accuracy: 0.9604 Validation accuracy: 0.9736
Model2 - Train accuracy: 0.9560 Validation accuracy: 0.9824
Model3 - Train accuracy: 0.8879 Validation accuracy: 0.8771

### Optimizers - SGD, RMSprop, Adam

```python
input_shape = 30
output_shape = 1
loss='binary_crossentropy', learning_rate=1e-3, batch_size=1, epochs=30

    model = Model()
    model.create_network([
        Dense(input_shape, 20, activation='relu'),
        Dense(20, 10, activation='relu'),
        Dense(10, 5, activation='relu'),
        Dense(5, output_shape, activation='sigmoid')
        ])

 model_list = [
             (model1, optimizers.SGD(learning_rate=1e-3)),
             (model2, optimizers.RMSprop(learning_rate=1e-3)),
             (model3, optimizers.Adam(learning_rate=1e-3)),
]
```

| optimizers                                                                                                           |
| -------------------------------------------------------------------------------------------------------------------- |
| ![optimizers](https://github.com/jmcheon/multilayer_perceptron/assets/40683323/c1a78d9d-1e9f-431a-b560-ee880e751350) |

Model1 - Train accuracy: 0.9318 Validation accuracy: 0.9473
Model2 - Train accuracy: 0.9340 Validation accuracy: 0.9649
Model2 - Train accuracy: 0.9582 Validation accuracy: 0.9561
