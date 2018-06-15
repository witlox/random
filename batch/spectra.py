# coding: utf-8
import matplotlib
matplotlib.use('Agg')

import os
import time
import datetime
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get some variables, or use defaults
parser = argparse.ArgumentParser(description="run keras for spectra analysis RNN")
parser.add_argument("-g", "--gpus", default=1, type=int, help="Number of GPUs to allocate (default: %(default)s)")
parser.add_argument("-i", "--input", default="sml.tsv", type=str, help="TSV file containing training data (default: %(default)s)")
parser.add_argument("-r", "--ratio", default=0.2, type=float, help="Ratio of the dataset to use as test (default: %(default)s)")
parser.add_argument("-t", "--test-input", default=None, type=str, help="TSV file containing evaluation data (default: %(default)s)")
parser.add_argument("-s", "--seed", default=42, type=int, help="Random seed for numpy (default: %(default)s)")
# softmax on single output will always normalize whatever comes in to 1.0.
parser.add_argument("-a", "--activation", default="sigmoid", type=str, help="Activation function to use (default: %(default)s)")
# this fixes the accuracy display, possibly more; you want to pass show_accuracy=True to model.fit()
parser.add_argument("-l", "--loss", default="binary_crossentropy", type=str, help="Loss function to use (default: %(default)s)")
parser.add_argument("-o", "--optimize", default="adam", type=str, help="Optimizer to use (default: %(default)s)")
parser.add_argument("-e", "--epochs", default=4, type=int, help="Number of epochs (default: %(default)s)")
parser.add_argument("-b", "--batch-size", default=64, type=int, help="Number batches (default: %(default)s)")
parser.add_argument("-w", "--write-to", default="outputs/sml", type=str, help="Directory to write to (overwrites it) (default: %(default)s)")
parser.add_argument("--confusion-matrix", dest="matrix", action="store_true", help="Generate confusion matrix (can crash the run)")
parser.set_defaults(matrix=False)

ns = parser.parse_args()

if not os.path.exists(ns.write_to):
    path = Path(ns.write_to)
    path.mkdir(parents=True)

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model

output_dimension = 1

start_dt = datetime.datetime.now()

start_time = time.time()

print("loading data sets ({0})".format(ns.input))
df = pd.read_csv(ns.input, "rb", delimiter="\t")

if ns.test_input:
    print("validation input specified ({0})".format(ns.test_input))
    validate = pd.read_csv(ns.test_input, "rb", delimiter="\t")

stop_time = time.time()
timing_load = stop_time - start_time

print("fixing our random ({0})".format(ns.seed))
np.random.seed(ns.seed)

print("selecting Inputs/Outputs")
input_cols = ['TIC','scan_time','v0','v1','v2','v3','v4','v5','v6','v7','v8','v9',
              'v10','v11','v12','v13','v14','v15','v16','v17','v18','v19',
              'v20','v21','v22','v23','v24','v25','v26','v27','v28','v29',
              'v30','v31','v32','v33','v34','v35','v36','v37','v38','v39',
              'v40','v41','v42','v43','v44','v45','v46','v47','v48','v49',
             ]
output_cols = ['cls']

start_time = time.time()

print("splitting taining")
train, test = train_test_split(df, test_size=ns.ratio)
print("building training")
x_train = train[input_cols].values
y_train = train[output_cols].values
print("building test")
x_test = test[input_cols].values
y_test = test[output_cols].values

if ns.test_input:
    print("building validate")
    x_val = validate[input_cols].values
    y_val = validate[output_cols].values

print("reshaping to correct dimensions")
x_train = np.reshape(x_train,(x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test,(x_test.shape[0], 1, x_test.shape[1]))

if ns.test_input:
    x_val = np.reshape(x_val,(x_val.shape[0], 1, x_val.shape[1]))

stop_time = time.time()
timing_shaping = stop_time - start_time

print("input training shape: {0}".format(x_train.shape))
print("output training shape: {0}".format(y_train.shape))

print("input testing shape: {0}".format(x_test.shape))
print("output testing shape: {0}".format(y_test.shape))

if ns.test_input:
    print("input validation shape: {0}".format(x_val.shape))
    print("output validation shape: {0}".format(y_val.shape))

start_time = time.time()

print("creating our model")
model = Sequential()

model.add(LSTM(output_dimension, input_shape=(1, len(input_cols))))

model.add(Dense(output_dimension, activation=ns.activation))

print("trying to set multi gpu stuff")
try:
    model = multi_gpu_model(model, gpus=ns.gpus)
    print("training on {0} GPU's".format(ns.gpus))
except:
    print("training on single CPU/GPU")

model.compile(loss=ns.loss, optimizer=ns.optimize, metrics=['accuracy'])

print(model.summary())

stop_time = time.time()
timing_modelling = stop_time - start_time

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=ns.epochs, batch_size=ns.batch_size)

stop_time = time.time()
timing_fitting = stop_time - start_time

print(hist.history)

loss = hist.history['loss']
acc = hist.history['acc']
xc = range(ns.epochs)

print("visualizing losses")
plt.figure(1,figsize=(7,5))
plt.plot(xc,loss)
plt.plot(xc,acc)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('loss vs accuracy')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])
plt.savefig("{0}/sml_loss_acc.png".format(ns.write_to))

if ns.matrix:
    print("confusion matrix")
    y_pred = model.predict_classes(x_test)
    print(y_pred)
    with open("{0}/sml_confusion.txt", "w") as confmat:
        confmat.write(y_pred)

start_time = time.time()

scores = model.evaluate(x_test, y_test)

stop_time = time.time()
timing_evaluating = stop_time - start_time

print("Accuracy: {0:.2f}".format(scores[1]*100))

print("serializing model to JSON")
model_json = model.to_json()
with open("{0}/sml_model.json".format(ns.write_to), "w") as json_file:
    json_file.write(model_json)
print("serializing weights to HDF5")
model.save_weights("{0}/sml_model.h5".format(ns.write_to))
print("Saved model to disk")

if ns.test_input:
    print("Running evaluation")
    scores_eval = model.evaluate(x_val, y_val)
    print("Explicit evaluation accuracy: {):/2f}".format(scores_eval[1]*100))

with open("{0}/timings{1:%H%M-%d%m%Y}.txt".format(ns.write_to, datetime.datetime.now()), "w") as jt:
    jt.write("started on {0}\n".format(start_dt))
    jt.write("------------------------------------------------\n")
    jt.write("load: {0} ms\n".format(timing_load*1000))
    jt.write("shaping: {0} ms\n".format(timing_shaping*1000))
    jt.write("modelling: {0} ms\n".format(timing_modelling*1000))
    jt.write("fitting: {0} ms\n".format(timing_fitting*1000))
    jt.write("evaluating: {0} ms\n".format(timing_evaluating*1000))
    jt.write("------------------------------------------------\n")
    jt.write("finished on {0}\n".format(datetime.datetime.now()))

