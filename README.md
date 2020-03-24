# Building and Deploying a Spark ML Model on the Cloud
## Table of Contents  

* [Introudction](#ab)  
* [Setting up instances on AWS](#ac) 
  * [EC2 instance setting](#ad)
  * [Software installation](#ae)

<a name = "ab"/>

## Introduction

This document describes the principal steps necessary for building a production grade deep learning model and deploy it on the cloud. Our focus is on the necessary hardware and software infrastructures for model deployment on the cloud, rather than finessing on building a perfect model. 

The deep learning model is about inferring textual similarity such as comparing similarity between pairs of sentences as in Quora question-pairs [[Kaggle](https://www.kaggle.com/c/quora-question-pairs)]. 

The model utilizes [Apache Spark](https://spark.apache.org/) for handling big-data and [Spark-NLP](https://github.com/JohnSnowLabs/spark-nlp) from [John Snow Labs](https://www.johnsnowlabs.com/) for creating ML pipelines in order to extract the feature vectors. Finally, [Keras](https://keras.io/) with [TensorFlow](https://www.tensorflow.org/) as backend is used to build the model. 

<a name ="ac"/>

## Setting up instances on AWS

<a name = "ad"/>

### - EC2 instance setting
 
<a name ="ae"/>

### - Software installation

