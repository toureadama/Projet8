#!/usr/bin/env python
# coding: utf-8

# ### I- Extraction des features avec RESNET50

# In[21]:


import os
import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import lit
from pyspark.sql.types import *
from sparkdl import DeepImageFeaturizer


# #### Spark configuration

# In[6]:


os.environ['PYSPARK_SUBMIT_ARGS'] = '-- packages com.amazonaws:aws-java-sdk:1.7.4,org.apache.hadoop:hadoop-aws:2.7.3 pyspark-shell'


# In[26]:


awsAccessKeyID='AKIARGO6ZPH2GYFMZPGI'
awsAccessSecretKey='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'


# In[27]:


conf = (
        SparkConf()
            .setAppName("pyspark_aws_project8")
            .set("spark.hadoop.fs.s3a.access.key", awsAccessKeyID)
            .set("spark.hadoop.fs.s3a.secret.key", awsAccessSecretKey)
            .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .set("spark.hadoop.fs.s3a..endpoint", "s3-eu-west-1.amazonaws.com")
            .set("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
            .set('spark.executor.extraJavaOptions','-Dcom.amazonaws.services.s3.enableV4=true')
            .set('spark.driver.extraJavaOptions','-Dcom.amazonaws.services.s3.enableV4=true')
            .set("spark.speculation", "false")
            .set("spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored", "true")
            .set("fs.s3a.experimental.input.fadvise", "random")
            .setIfMissing("spark.master", "local")
    )


# In[1]:


sc=SparkContext(conf=conf)


# In[2]:


spark=SparkSession(sc)


# In[20]:


datasets_echant_path = 's3a:\\my-project8-image\\donnees_image_echant\\fruits-360_dataset\\fruits-360\\Training\\'


# In[ ]:


# Create an empty spark dataframe
emp_RDD = spark.sparkContext.emptyRDD()
 
# Defining the schema of the DataFrame
columns1 = StructType([StructField('image',
                                   StructType([
                                       StructField("origin", StringType(), True),
                                       StructField("height", IntegerType(), True),
                                       StructField("width", IntegerType(), True),
                                       StructField("nChannels", IntegerType(), True),
                                       StructField("mode", IntegerType(), True),
                                       StructField("data", BinaryType(), True),
                                       ]), True,),
                       StructField('label', StringType(), False)])
 
# Creating an empty DataFrame
empty_df = spark.createDataFrame(data=emp_RDD,
                                         schema=columns1)


# In[ ]:


# Download all pictures with labels

df = empty_df
for dirname, _, _ in os.walk(datasets_echant_path):
    if dirname != datasets_echant_path:
        df_img = spark.read.format("image").load(dirname).withColumn("label", lit(dirname.split('/')[-1]))
        df = df.union(df_img)


# In[ ]:


# Extract features from images with Resnet50
featurizer = DeepImageFeaturizer(inputCol="image",
                                 outputCol="features",
                                 modelName="ResNet50")

df_feat = spark.createDataFrame(featurizer.transform(df.drop('label'))).WithColum('label',df.label)


# ### Réduction de dimension par PCA

# In[ ]:


from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler


# In[ ]:


cols = df_feat.drop('label').columns


# In[ ]:


# Create vector columns
assembler = VectorAssembler(inputCols=cols, outputCol = 'features')
output_dat = assembler.transform(df_feat).select('label', 'features')


# In[ ]:


# Apply PCA
pca = PCA(k=88, inputCol = "features", outputCol="pcaFeatures")

model = pca.fit(output_dat)
pca_features = model.transform(output_dat).select('label', df_feat.label)


# ### Stocker le fichier de sortie sur S3

# In[ ]:


# Define the s3 destination path
s3_dest_path = "s3a://sortieimagestraitees"
print("s3 destination path "+s3_dest_path)


# In[ ]:


# Write the data as csv
pcaFeatureCsvPath = s3_dest_path
pca_features.write.mode("overwrite").format("csv").save(pcaFeatureCsvPath)


# #### Arrêter la session Spark

# In[3]:


spark.stop()


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:




