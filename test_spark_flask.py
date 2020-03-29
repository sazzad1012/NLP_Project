import os
import sys
import string
import pandas as pd
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
import pyspark
import sparknlp
from sparknlp import *
from sparknlp.annotator import Lemmatizer, Stemmer, Tokenizer, Normalizer
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.embeddings import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec
from pyspark.ml import feature as spark_ft

#Directory where the model is saved
model_dir = '/home/ubuntu/...../'

#Create Spark session
spark = SparkSession.builder \
 .master('local[1]') \
 .appName('Spark NLP') \
 .config('spark.driver.memory', '12g') \
 .config('spark.executor.memory', '12g') \
 .config('spark.jars.packages', 'JohnSnowLabs:spark-nlp:2.4.0') \
 .getOrCreate()

sql = SQLContext(spark)

import keras
from keras.layers import concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, LSTM, Lambda
from tensorflow.keras import Input
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

#Load the model
new_model = load_model(os.path.join(model_dir, 'test_result.h5'))

#Data pipeline tokenizing the input questions
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

    return processed1

# Feature extraction
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
   
#Flask app
app = Flask(__name__)
@app.route('/')
@app.route('/form-example', methods=['GET', 'POST']) #allow both GET and POST requests
def form_example():
    if request.method == 'POST':  #this block is only entered when the form is submitted
        texta = request.form.get('texta')
        textb = request.form['textb']
#Read the inputs for further processing before feeding into the model input
        test_list = [(texta, textb)]
        dfgiven = sql.createDataFrame(test_list, ['question1', 'question2'])
        test = build_data(dfgiven)
        test_qA, test_qB = feature_extract(test)
 # Load the model
        new_model = load_model(os.path.join(model_dir, 'test_result.h5'))
        new_prediction = new_model.predict([test_qA,test_qB])
        if new_prediction >0.5:
            res = 'Yes'
        else:
            res = 'No'
        return '''<h1>Do the questions have similar meaning?  {}</h1>'''.format(res)

    return '''<form method="POST">
                  Question1: <input type="text" name="texta"><br>
                  Question2: <input type="text" name="textb"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''
   
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
    
