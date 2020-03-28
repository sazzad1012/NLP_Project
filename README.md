# Building and Deploying a Spark ML Model on the Cloud
## Table of Contents  

* [Introudction](#ab)  
* [Computing resources on AWS](#ac) 
  * [Software installation](#ae)
* [Deep learning model](#af)
  * [Model pipeline](#ag)
  * [Keras model](#ah)
* [Deployment](#ai)
* [Conclusion](#aj)

<a name = "ab"/>

## 1. Introduction
This document describes the principal steps necessary for building a deep learning model and deploying it on the cloud, [Amazon Web Services](https://aws.amazon.com/). Our focus is on provisioning the necessary hardware and software infrastructures for model deployment on the cloud. 

The deep learning model is about inferring textual similarity such as comparing sentences as in Quora question-pairs [[Kaggle](https://www.kaggle.com/c/quora-question-pairs)]. 

The model utilizes [Apache Spark](https://spark.apache.org/) for handling big-data and [Spark-NLP](https://github.com/JohnSnowLabs/spark-nlp) developed by [John Snow Labs](https://www.johnsnowlabs.com/) for creating the ML pipelines in order to extract the feature vectors. Finally, [Keras](https://keras.io/) with [TensorFlow](https://www.tensorflow.org/) as backend is used for building the model. 
<a name ="ac"/>

## 2. Computing resources on AWS
The first step is provisoning for adequate cloud computing reosurces, for which our choice is [AWS](https://aws.amazon.com/). Amazon Elastic Compute Cloud, [**Amazon EC2**](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Instances.html), provides secure, resizable compute capacity in the cloud. We created two linux-based ``Ubuntu 18.04`` EC2 instances following the excellent [documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance). We used ``t3.2xlarge`` that has ``8 vCPU``, ``64GB`` memeory with ``120GB`` storage capacity.  
<a name ="ae"/>

### 2a. Software installation
One of the trickst part is to setup the compute resurces with appropriate softwares. First, we need to install ```Spark``` on each EC2 instance and then we install the necessary ```Python``` packages as described below. 

After, intsalling the necessary softwares, we used ```PyCharm``` profeesional edition from [JetBrains](https://www.jetbrains.com/pycharm/) with its remote-interpreter capability to connect with the EC2 intstances and for code development. 

#### --Installation of Spark and Spark-NLP
There are well-documented online resources for installing ```Spark``` on EC2 instances ([1](https://github.com/tkachuksergiy/aws-spark-nlp), [2](https://computingforgeeks.com/how-to-install-apache-spark-on-ubuntu-debian/), [3](https://blog.insightdatascience.com/simply-install-spark-cluster-mode-341843a52b88)). The basic step requires installing ```Java```, downloading ```Spark 2.4.4```, and setting-up the master and the slave node. In our case, there is one master and one slave node, since we chose to have two EC2 instances. One crucial step is updating the ```.bashrc``` configuration file:
```bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export JRE_HOME=$JAVA_HOME/jre
export PATH=$PATH:$JAVA_HOME/bin:$JAVA_HOME/jre/bin
export SPARK_HOME=/usr/local/spark-2.4.0-bin-hadoop2.7
export PATH=$PATH:$SPARK_HOME/bin
export PYSPARK_PYTHON=python3
export SYSTEMML_HOME=/usr/local/systemml-1.2.0-bin
```
Once ```Spark (PySpark)``` is properly installed, [Spark NLP 2.4.0]((https://github.com/JohnSnowLabs/spark-nlp)) needs to be installed using ```pip```- the ```Python``` package installer. 

#### --Intsallation of Python packages
Lastly, we need to install ```Keras``` and ```TensorFlow```([4](https://www.pyimagesearch.com/2019/01/30/ubuntu-18-04-install-tensorflow-and-keras-for-deep-learning/)) along with ```Python``` packages for natural language processing, qunatittaive data analyses and plotting such as ```NLTK```, ```NumPy```, ```Pandas```, ```Seaborn```. 
<a name = "af"/>

## 3. Deep learning model
The NLP deep learning model for comapring question-pairs has two components. The first is about data-pipelining with feature extraction. The features after approproiate vectorization are fed into a LSTM model the output of which are then used for model evaluation. The ```Spark``` nodes are appropriately set (in the example below the master node is set to local) to read the data-file:
```python
spark = SparkSession.builder \
 .master('local[4]') \
 .appName('Spark NLP') \
 .config('spark.driver.memory', '12g') \
 .config('spark.executor.memory', '12g') \
 .config('spark.jars.packages', 'JohnSnowLabs:spark-nlp:2.4.0') \
 .getOrCreate()
```
Then, the dataframe, after being read the data file, is split into two dataframes for tarining and validation:
```python
sql = SQLContext(spark)
dfgiven = sql.read.csv(f'{train_dir}{file_name}', header=True, inferSchema=True, escape = '\"')
df1,df2 = dfgiven.randomSplit([0.50, 0.50],seed=1234)
```
<a name ="ag"/>

### 3a. Model pipeline
A helper function makes it easy for creating the data pipeline and extracting the features. First, the questions are tokenized and assembeld into a pipleine using Spark-NLP (see ```xxx.py```): 
```pyton
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
    .......
    p_pipeline = Pipeline(stages=[document_assembler1, tokenizer1, finisher1, \
                                  document_assembler2, tokenizer2, finisher2])
    return ...
```
For extracting the features, first stop-words are removed and then the word-tokens are converted into feature vectors (see ```yyy.py```)  using [Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) as implemented in ```Spark NLP```:
```python
def feature_extract(train_t):
    stopWords = spark_ft.StopWordsRemover.loadDefaultStopWords('english')
    sw_remover1 = spark_ft.StopWordsRemover(inputCol='ntokens1', outputCol='clean_tokens1', stopWords=stopWords)
    text2vec1 = spark_ft.Word2Vec(
        vectorSize=50, minCount=1, seed=123,
        inputCol='ntokens1', outputCol='text_vec1',
        windowSize=1, maxSentenceLength=100)
    assembler1 = spark_ft.VectorAssembler(inputCols=['text_vec1'], outputCol='features1')
    ......
    feature_pipeline = Pipeline(stages=[sw_remover1, text2vec1, assembler1, sw_remover2, text2vec2, assembler2])
    feature_model = feature_pipeline.fit(train_t)
    ......
    return .... 
```
<a name ="ah"/>

### 3b. Keras model
The Keras deep learning model has a [simaese recurrent](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12195/12023). This is a bi-lstm mode, where each input question is fed into a lastm model then the output from the two lstm are comapred using a exponential cpmarator. An schematic of the model architecture is:

![Image1](https://github.com/sazzad1012/NLP_Project/blob/master/Blstm.png)

The model uses a binary cross-entrypy loss function and a logistic output, which gives the probbaility of whether a given question-pair is similar or dissimilar.
```python
lstm = layers.LSTM(n_hidden, unit_forget_bias=True, kernel_initializer='he_normal',\
                            kernel_regularizer='l2', name='lstm_layer')
left_input = Input(shape=(None, input_dim), name='input_1')
left_output = lstm(left_input)
right_input = Input(shape=(None, input_dim), name='input_2')
right_output = lstm(right_input)
l1_norm = lambda x: 1 - K.abs(x[0] - x[1])
merged = layers.Lambda(function=l1_norm, output_shape=lambda x: x[0], \
                                  name='L1_distance')([left_output, right_output])
predictions = layers.Dense(1, activation='relu', name='Similarity_layer')(merged)
model = Model([left_input, right_input], predictions)
.....
```
Here is the model summary:

![Image3](https://github.com/sazzad1012/NLP_Project/blob/master/ModelSummary1.png)

The model accuracy on the validation data set is:

![Image4](https://github.com/sazzad1012/NLP_Project/blob/master/Acc.png)

The model is then serialized and saved as a ```hdf``` file.
<a name = "ai"/>

## 4. Deployment
The model is deployed with a ```Flask``` web application that interfaces with ```Nginx``` web-server. The technology stack is depicted as follows:

![Image2](https://github.com/sazzad1012/NLP_Project/blob/master/FlaskS.png)

Here, ```Gunicorn``` is a ```Python``` based ```WSGI http``` server that interfaces between ```Flask``` and ```Nginx.``` We followed two excellent tutorials [4](https://medium.com/hackernoon/a-guide-to-scaling-machine-learning-models-in-production-aa8831163846) and [5](https://www.jashds.com/blog/host-your-deep-learning-model-on-aws-free). After downloading and setting up ```Nginx``` and ```Gunicorn```, a ```Flask``` application is writtent. Finally, the ```systemmd``` service needs to be properly configured to ensure smooth running of the web application.

Below is the ```Flask``` application (see ```zzz.py```): 
```python
app = Flask(__name__)
@app.route('/')
@app.route('/form-example', methods=['GET', 'POST']) 
def form_example():
    if request.method == 'POST': 
        texta = request.form.get('texta')
        textb = request.form['textb']
        test_list = [(texta, textb)]
        dfgiven = sql.createDataFrame(test_list, ['question1', 'question2'])
        test = build_data(dfgiven)
        test_qA, test_qB = feature_extract(test)
        new_prediction = new_model.predict([test_qA,test_qB])
        return jsonify(str(new_prediction))

    return '''<form method="POST">
                  Question1: <input type="text" name="texta"><br>
                  Question2: <input type="text" name="textb"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
```
Upon enetering two query questions, the program spits out a result:

![Image4](https://github.com/sazzad1012/NLP_Project/blob/master/Predict.png)

