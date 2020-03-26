# Building and Deploying a Spark ML Model on the Cloud
## Table of Contents  

* [Introudction](#ab)  
* [Computing resources on AWS](#ac) 
  * [Software installation](#ae)
* [Deep learning model](#af)


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
#### --Installation of Spark and Spark-NLP
There are excellent online resources for installing Spoark on EC2 instances ([1](https://github.com/tkachuksergiy/aws-spark-nlp), [2](https://computingforgeeks.com/how-to-install-apache-spark-on-ubuntu-debian/), [3](https://blog.insightdatascience.com/simply-install-spark-cluster-mode-341843a52b88)). This requires installing ```Java```, downloading ```Spark 2.4.4```, settingup the master and the slave node. In our case, there is one master and one slave node, since we chose to have two E2 instances. One crucial step is updating the ```.bashrc``` configuration file:

```bash
export SPARK_HOME=/usr/local/spark-2.3.2-bin-hadoop2.7
export PATH=$PATH:$SPARK_HOME/bin
export PYSPARK_PYTHON=python3
. ~/.profile

sudo chown -R ubuntu $SPARK_HOME
```
Once ```Spark (PySpark)``` is properly installed, [Spark NLP 2.4.0]((https://github.com/JohnSnowLabs/spark-nlp)) needs to be installed using ```pip```- the ```Python``` package installer. 

#### --Intsallation of Python packages
Finally, we need to install ```Keras``` and ```TensorFlow```([4](https://www.pyimagesearch.com/2019/01/30/ubuntu-18-04-install-tensorflow-and-keras-for-deep-learning/)) along with ```Python``` packages such as ```NLTK```, ```NumPy```, and ```Pandas```. 

<a name = "af"/>

## Deep learning model

```python
spark = SparkSession.builder \
 .master('local[4]') \
 .appName('Spark NLP') \
 .config('spark.driver.memory', '12g') \
 .config('spark.executor.memory', '12g') \
 .config('spark.jars.packages', 'JohnSnowLabs:spark-nlp:2.4.0') \
 .getOrCreate()
```

Then read the data file and split it into two dataframes for tarining and validation:

```python
sql = SQLContext(spark)
dfgiven = sql.read.csv(f'{train_dir}{file_name}', header=True, inferSchema=True, escape = '\"')
df1,df2 = dfgiven.randomSplit([0.50, 0.50],seed=1234)
```

Then write a helper function creating a dartapipeline for feature extraction.
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
```

Then extracting the features:
<div class="text-white bg-gray-dark mb-2">
```python
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
    ```
deep learning model

An schematic of the model is:
![Image1](https://github.com/sazzad1012/NLP_Project/blob/master/test.003.png)


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
#predictions = layers.Dense(1, activation='sigmoid', name='Similarity_layer')(merged)
predictions = layers.Dense(1, activation='relu', name='Similarity_layer')(merged)
model = Model([left_input, right_input], predictions)

optimizer = Adadelta()
#optimizer = Adadelta(learning_rate=1.05, rho=0.85)
#optimizer = Adam(lr=0.001)
model.compile(loss = 'mse', optimizer = optimizer, metrics=['accuracy'])
#model.compile(loss = tf.keras.losses.BinaryCrossentropy(), optimizer = optimizer, metrics=['accuracy'])
history = model.fit([train_qA, train_qB], train_scores, batch_size=64, nb_epoch=15, validation_data=([val_qA, val_qB], val_scores))
#model.save('/home/ubuntu/ML_NLP/test_result1.h5')
```

```python
app = Flask(__name__)
@app.route('/')
@app.route('/form-example', methods=['GET', 'POST']) #allow both GET and POST requests
def form_example():
    if request.method == 'POST':  #this block is only entered when the form is submitted
        texta = request.form.get('texta')
        textb = request.form['textb']

        test_list = [(texta, textb)]
        dfgiven = sql.createDataFrame(test_list, ['question1', 'question2'])
        test = build_data(dfgiven)
        test_qA, test_qB = feature_extract(test)
        new_prediction = new_model.predict([test_qA,test_qB])
        return jsonify(str(new_prediction))
#        return '''<h1>The language value is: {}</h1>'''.format(texta)
#                  <h1>The framework value is: {}</h1>'''.format(texta, textb)

    return '''<form method="POST">
                  Question1: <input type="text" name="texta"><br>
                  Question2: <input type="text" name="textb"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
```

