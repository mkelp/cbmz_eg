#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt  # Python standard library datetime  module
import csv




def _stats(a):
    """ calc_stats calculates the mean and standard deviation of numpy array a."""
    a_std = a.std(axis=0)
    # We don't want to be solving for numerical noise.
    a_std[a_std < 1.e-6] = 1
    return _twod(a.mean(axis=0)), _twod(a_std)

def _twod(a):
    """ twod converts a 1-d array to 2-d """
    if len(a.shape) == 1:
        a = a.reshape((1, a.shape[0]))
    return a

def decode_func(num_conc, num_met):
    """ decode decodes a single TFRecord example with input concentrataion
    and meteorology and output delta concentration information. """
    def _decode(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'conc': tf.FixedLenFeature([num_conc], tf.float32),
                'met': tf.FixedLenFeature([num_met], tf.float32),
                'delta': tf.FixedLenFeature([num_conc], tf.float32),
            })
        return {"conc":features["conc"], "met": features["met"]}, features["delta"]
    return _decode


n_conc_vars = 77
n_met_vars = 4
n_samples = 1000000

dataset = tf.data.TFRecordDataset("data/dt60_diurnal_10000080r_t0.tfrecords")
dataset = dataset.apply(tf.contrib.data.map_and_batch(
    decode_func(n_conc_vars, n_met_vars),n_samples))
inputs, input_delta = dataset.make_one_shot_iterator().get_next()
sess = tf.Session()
with sess.as_default():
    conc_, met_, delta_ = sess.run([inputs["conc"], inputs["met"], input_delta])
    conc_mean, conc_std = _stats(conc_)
    met_mean, met_std = _stats(met_)
    delta_mean, delta_std = _stats(delta_)

print(delta_mean)
