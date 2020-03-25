# Building and Deploying a Spark ML Model on the Cloud
## Table of Contents  

* [Introudction](#ab)  
* [Computing resources on AWS](#ac) 
  * [EC2 instance](#ad)
  * [Software installation](#ae)

<a name = "ab"/>

## Introduction

This document describes the principal steps necessary for building a production grade deep learning model and deploy it on the cloud, [Amazon Web Services](https://aws.amazon.com/). Our focus is on the necessary hardware and software infrastructures for model deployment on the cloud, rather than finessing on building a perfect model. 

The deep learning model is about inferring textual similarity such as comparing similarity between pairs of sentences as in Quora question-pairs [[Kaggle](https://www.kaggle.com/c/quora-question-pairs)]. 

The model utilizes [Apache Spark](https://spark.apache.org/) for handling big-data and [Spark-NLP](https://github.com/JohnSnowLabs/spark-nlp) from [John Snow Labs](https://www.johnsnowlabs.com/) for creating ML pipelines in order to extract the feature vectors. Finally, [Keras](https://keras.io/) with [TensorFlow](https://www.tensorflow.org/) as backend is used to build the model. 

<a name ="ac"/>

## Computing resources on AWS

The first step is provisoning for adequate cloud computing reosurces, for which an excellent choice is [AWS](https://aws.amazon.com/). Amazon Elastic Compute Cloud (**[Amazon EC2]**)(https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Instances.html) provides secure, resizable compute capacity in the cloud. We created two linux-based (**Ubuntu 18.04**) EC2 instances following the excellent [documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance). We also used ``p3.2xlarge`` that has 1 GPU and 32 vCPU.  
 
<a name = "ad"/>

### - EC2 instance setting
 
<a name ="ae"/>

### - Software installation

