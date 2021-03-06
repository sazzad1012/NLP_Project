#Import packages
import os
import sys
import string
import pandas as pd
import numpy as np
import pandas as pd
import pyspark
import sparknlp
from sparknlp import *
from sparknlp.annotator import Lemmatizer, Stemmer, Tokenizer, Normalizer
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.embeddings import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec
from pyspark.ml import feature as spark_ft

#Data path and file name
train_dir = 'Path_to_training_data/'
file_name ='train1.csv'

#Create Spark session
spark = SparkSession.builder \
 .master('local[4]') \
 .appName('Spark NLP') \
 .config('spark.driver.memory', '12g') \
 .config('spark.executor.memory', '12g') \
 .config('spark.jars.packages', 'JohnSnowLabs:spark-nlp:2.4.0') \
 .getOrCreate()

#Read data file
sql = SQLContext(spark)
dfgiven = sql.read.csv(f'{train_dir}{file_name}', header=True, inferSchema=True, escape = '\"')
df1,df2 = dfgiven.randomSplit([0.50, 0.50],seed=1234) #training and validation set
df1 = sql.createDataFrame(df1.head(500), df1.schema)#Just taking the 1st 500 samples
df1 = df1.na.drop()
df2 = sql.createDataFrame(df2.head(500), df2.schema)
df2 = df2.na.drop()

#Helper function for tokenizing the text using Spark-NLP (in this case two question columns: Question 1 & Question 2)
def build_data(df):
    document_assembler1 = DocumentAssembler() \
        .setInputCol('question1').setOutputCol('document1')

    tokenizer1 = Tokenizer() \
        .setInputCols(['document1']) \
        .setOutputCol('token1')

    finisher1 = Finisher() \
        .setInputCols(['token1']) \
        .setOutputCols(['ntokens1']) \
        .setOutputAsArray(True) \
        .setCleanAnnotations(True)

    document_assembler2 = DocumentAssembler() \
        .setInputCol('question2').setOutputCol('document2')

    tokenizer2 = Tokenizer() \
        .setInputCols(['document2']) \
        .setOutputCol('token2')

    finisher2 = Finisher() \
        .setInputCols(['token2']) \
        .setOutputCols(['ntokens2']) \
        .setOutputAsArray(True) \
        .setCleanAnnotations(True)

    p_pipeline = Pipeline(stages=[document_assembler1, tokenizer1, finisher1, \
                                  document_assembler2, tokenizer2, finisher2])
    p_model = p_pipeline.fit(df)
    processed1 = p_model.transform(df)
    label1 = processed1.select('is_duplicate').collect()
    label_array1 = np.array(label1)
    label_array1 = label_array1.astype(np.int)

    return processed1, label_array1

#Helper function for feature extraction and vecotorization using word2vec (Spark-NLP)
def feature_extract(train_t):
    stopWords = spark_ft.StopWordsRemover.loadDefaultStopWords('english')

    sw_remover1 = spark_ft.StopWordsRemover(inputCol='ntokens1', outputCol='clean_tokens1', stopWords=stopWords)

    text2vec1 = spark_ft.Word2Vec(
        vectorSize=50, minCount=1, seed=123,
        inputCol='ntokens1', outputCol='text_vec1',
        windowSize=1, maxSentenceLength=100)

    assembler1 = spark_ft.VectorAssembler(inputCols=['text_vec1'], outputCol='features1')

    sw_remover2 = spark_ft.StopWordsRemover(inputCol='ntokens2', outputCol='clean_tokens2', stopWords=stopWords)

    text2vec2 = spark_ft.Word2Vec(
        vectorSize=50, minCount=1, seed=123,
        inputCol='ntokens2', outputCol='text_vec2',
        windowSize=1, maxSentenceLength=100)

    assembler2 = spark_ft.VectorAssembler(inputCols=['text_vec2'], outputCol='features2')

    feature_pipeline = Pipeline(stages=[sw_remover1, text2vec1, assembler1, sw_remover2, text2vec2, assembler2])

    feature_model = feature_pipeline.fit(train_t)

    train_featurized = feature_model.transform(train_t).persist()
    tA = train_featurized.select('text_vec1').collect()
    tA_array = np.array(tA)
    tB = train_featurized.select('text_vec2').collect()
    tB_array = np.array(tB)

    return tA_array, tB_array
    
#Now convert the training and validation data into vectors that can be used to inpiut for the DL model
train, train_scores = build_data(df1)
val, val_scores = build_data(df2)
train_qA, train_qB = feature_extract(train)
val_qA, val_qB = feature_extract(val)

#Keras & TensorFlow imports
from keras.layers import concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, LSTM, Lambda
from tensorflow.keras import Input
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

#Keras hyperparameters
input_dim = 50
h_units = 100
n_hidden = 50

#Create lstm layers, one for each question 
lstm = layers.LSTM(n_hidden, unit_forget_bias=True, kernel_initializer='he_normal',\
                            kernel_regularizer='l2', name='lstm_layer')
left_input = Input(shape=(None, input_dim), name='input_1')
left_output = lstm(left_input)
right_input = Input(shape=(None, input_dim), name='input_2')
right_output = lstm(right_input)
l1_norm = lambda x: 1 - K.abs(x[0] - x[1])
merged = layers.Lambda(function=l1_norm, output_shape=lambda x: x[0], \
                                  name='L1_distance')([left_output, right_output])
predictions = layers.Dense(1, activation='sigmoid', name='Similarity_layer')(merged)
#Two lstm layers ar merged before making the prediction
model = Model([left_input, right_input], predictions)
optimizer = Adadelta()

#Model compilation and etc...
model.compile(loss = tf.keras.losses.BinaryCrossentropy(), optimizer = optimizer, metrics=['accuracy'])
history = model.fit([train_qA, train_qB], train_scores, batch_size=64, nb_epoch=15, validation_data=([val_qA, val_qB], val_scores))

#Saving the model
model.save('path_to_save_folder/test_result1.h5')
