# Building and Deploying a Spark ML Model on the Cloud
## Table of Contents  

* [Introudction](#ab)  
* [Computing resources on AWS](#ac) 
  * [Software installation](#ae)

<a name = "ab"/>

## Introduction
This document describes the principal steps necessary for building a production grade deep learning model and deploy it on the cloud, [Amazon Web Services](https://aws.amazon.com/). Our focus is on the necessary hardware and software infrastructures for model deployment on the cloud, rather than finessing on building a perfect model. 

The deep learning model is about inferring textual similarity such as comparing similarity between pairs of sentences as in Quora question-pairs [[Kaggle](https://www.kaggle.com/c/quora-question-pairs)]. 

The model utilizes [Apache Spark](https://spark.apache.org/) for handling big-data and [Spark-NLP](https://github.com/JohnSnowLabs/spark-nlp) from [John Snow Labs](https://www.johnsnowlabs.com/) for creating ML pipelines in order to extract the feature vectors. Finally, [Keras](https://keras.io/) with [TensorFlow](https://www.tensorflow.org/) as backend is used to build the model. 

<a name ="ac"/>

## Computing resources on AWS
The first step is provisoning for adequate cloud computing reosurces, for which an excellent choice is [AWS](https://aws.amazon.com/). Amazon Elastic Compute Cloud, [**Amazon EC2**](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Instances.html), provides secure, resizable compute capacity in the cloud. We created two linux-based ``Ubuntu 18.04`` EC2 instances following the excellent [documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance). We also used ``p3.2xlarge`` that has ``1 GPU``, ``32 vCPU``, ``64GB`` memeory with ``120GB`` storage capacity.  
 
<a name ="ae"/>

### -Software installation
First, we need to install Spark on each EC2 instance. Then, we will install the other spackages as described below.
#### --Spark installation
There are excellent online resources for installing Spoark on EC2 instances ([1](https://github.com/tkachuksergiy/aws-spark-nlp), [2](https://computingforgeeks.com/how-to-install-apache-spark-on-ubuntu-debian/), [3](https://blog.insightdatascience.com/simply-install-spark-cluster-mode-341843a52b88)). This requires installing ```Java```, downloading ```Spark 2.4.4```, settingup the master and the slave node. In our case, there is one master and one slave node, since we chose to have two E2 instances. One crucial step is updating the ```.bashrc``` configuration file:

```bash
export SPARK_HOME=/usr/local/spark-2.3.2-bin-hadoop2.7
export PATH=$PATH:$SPARK_HOME/bin
export PYSPARK_PYTHON=python3
. ~/.profile

sudo chown -R ubuntu $SPARK_HOME
```
Once ```Spark (PySpark)``` is properly installed, [Spark NLP 2.4.0]((https://github.com/JohnSnowLabs/spark-nlp)) needs to be installed using ```pip```- the ```Python``` package installer. 

#### --Intsalling Python packages
Fionally, we need to install ```Keras``` and ```TensorFlow```([4](https://www.pyimagesearch.com/2019/01/30/ubuntu-18-04-install-tensorflow-and-keras-for-deep-learning/)) along with ```Python``` packages for ```NLTK```, ```NumPy```, and ```Pandas```. 

