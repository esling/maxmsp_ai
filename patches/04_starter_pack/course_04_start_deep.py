#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 12:16:11 2023

@author: esling
"""

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import io
import csv

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_map_csv = io.StringIO(class_map_csv_text)
  class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
  class_names = class_names[1:]  # Skip CSV header
  return class_names

# Create our silence chunk
sr = 16000
len_sig = 3
silence = np.zeros(len_sig * sr, dtype=np.float32)
# Import the model
model = hub.load('https://tfhub.dev/google/yamnet/1')
# Perform classification
scores, embeddings, spectro = model(silence)

# Create class names associations
class_map_path = model.class_map_path().numpy()
# Final class names vector
class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))

# Get our overall class
overall_class = class_names[scores.numpy().mean(axis=0).argmax()]
print(overall_class)