import tensorflow as tf

coughdetector = tf.saved_model.load(r'./Models/coughdetector')
covidcough = tf.saved_model.load(r'./Models/covidcough')