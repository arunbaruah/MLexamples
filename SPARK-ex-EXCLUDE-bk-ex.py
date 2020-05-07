from pyspark.sql.dataframe import DataFrame
sax = spark._jvm.com.ralib.notebook.spark.SAXSparkSession.getSession(spark._jsparkSession)

ds_Churn_Modelling = DataFrame(sax.getDataSet("Churn_Modelling"), spark)

#imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler

#data preparation
ds_Churn_Modelling = ds_Churn_Modelling.drop('RowNumber','Gender', 'CustomerId', 'Surname', 'Geography')

assembler = VectorAssembler(
    inputCols=['CreditScore','Age','Tenure','Balance','NumOfProducts', 'HasCrCard',
                                               'IsActiveMember','EstimatedSalary'],
    outputCol="features")

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

#model preparation

#create Logistic Regression object
LR = LogisticRegression(labelCol='Exited', featuresCol='scaledFeatures', predictionCol ='Prediction')

#create pipeline
pipeline = Pipeline(stages=[assembler, scaler, LR])

#train model
model = pipeline.fit(ds_Churn_Modelling)

#make prediction
predictions = model.transform(ds_Churn_Modelling)
predictions.select('Prediction', 'Exited').show()

#save the model in hdfs.
#Note - you need a valid hdfs location with livy permission
model.write().overwrite().save('/LR_chrun_modelling_pyspark')

