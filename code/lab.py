import sys
import os
from pyspark.sql.functions import *
import pyspark.sql.functions as psf
from pyspark.sql.window import Window as W
from pyspark.sql.functions import col,max,lit
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import isnan, when, count, col, desc
from numpy import nan
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import substring
spark = SparkSession.builder.appName('Sri_Test_For_Paytm_labs').getOrCreate()

df_countrylist = spark.read.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.csv('file:///C:/Users/<username>/Desktop/paytm_labs/countrylist.csv')

df_stationlist = spark.read.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.csv('file:///C:/Users/<username>/Desktop/paytm_labs/stationlist.csv')

df_stationlist_countrylist = df_stationlist.join(df_countrylist, df_stationlist['COUNTRY_ABBR'] == df_countrylist['COUNTRY_ABBR']).select('*')

dataset = spark.read.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.csv('file:///C:/Users/<username>/Desktop/paytm_labs/data/2019/*.csv')

#for handling the missing data
dataset_new = dataset.withColumn('TEMP_new',when(col('TEMP')==9999.9, float("nan")))\
.withColumn('DEWP_new',when(col('DEWP')==9999.9, float("nan")))\
.withColumn('SLP_new',when(col('SLP')==9999.9, float("nan")))\
.withColumn('STP_new',when(col('STP')==9999.9, float("nan")))\
.withColumn('VISIB_new',when(col('VISIB')==999.9, float("nan")))\
.withColumn('WDSP_new',when(col('WDSP')==999.9, float("nan")))\
.withColumn('MXSPD_new',when(col('MXSPD')==999.9, float("nan")))\
.withColumn('GUST_new',when(col('GUST')==999.9, float("nan")))\
.withColumn('MAX_new',when(col('MAX')==9999.9, float("nan")))\
.withColumn('MIN_new',when(col('MIN')==9999.9, float("nan")))\
.withColumn('PRCP_new',when(col('PRCP')==99.9, float("nan")))\
.withColumn('SNDP_new',when(col('SNDP')==999.9, float("nan")))

#Calling the imputer function
imputer = Imputer(strategy='mean', missingValue=nan)
imputer.setInputCols(['TEMP_new','DEWP_new','SLP_new','STP_new','VISIB_new' ,'WDSP_new','MXSPD_new','GUST_new','MAX_new','MIN_new','PRCP_new','SNDP_new'])
imputer.setOutputCols(['TEMP_impu','DEWP_impu','SLP_impu','STP_impu','VISIB_impu' ,'WDSP_impu','MXSPD_impu','GUST_impu','MAX_impu','MIN_impu','PRCP_impu','SNDP_impu'])

#fitting and transforming model
model = imputer.fit(dataset_new)
clean_dataset = model.transform(dataset_new)

clean_dataset=clean_dataset.select('STN---','WBAN','YEARMODA','TEMP_impu','DEWP_impu','SLP_impu','STP_impu','VISIB_impu',\
                       'WDSP_impu','MXSPD_impu','GUST_impu','MAX_impu','MIN_impu','PRCP_impu','SNDP_impu','FRSHTT')

#Final Join with Country
final_dataset=clean_dataset.join(df_stationlist_countrylist, clean_dataset['STN---']==df_stationlist_countrylist['STN_NO']).select('STN---','WBAN','YEARMODA','TEMP_impu','DEWP_impu','SLP_impu','STP_impu','VISIB_impu',\
                       'WDSP_impu','MXSPD_impu','GUST_impu','MAX_impu','MIN_impu','PRCP_impu','SNDP_impu','FRSHTT','COUNTRY_FULL')
					   

#	
print('Running the query for QUESTION #1...........')
		
df_Question_1 = spark.sql('''with avg_temp_table as (
select distinct COUNTRY_FULL,substr(YEARMODA,1,4) YEAR,
avg(TEMP_impu) over(partition by COUNTRY_FULL,substring(YEARMODA,1,4)) avg_of_country_temprature
from final_dataset_table)
select COUNTRY_FULL, YEAR, avg_of_country_temprature from (select COUNTRY_FULL, YEAR, avg_of_country_temprature,
rank() over(partition by year order by avg_of_country_temprature desc) as rnk
from avg_temp_table)
where rnk=1
''')

df_Question_1.show()

df_Question_3 = spark.sql('''with avg_windspeed_table as (
select distinct COUNTRY_FULL,substr(YEARMODA,1,4) YEAR,
avg(WDSP_impu) over(partition by COUNTRY_FULL,substring(YEARMODA,1,4)) avg_of_country_windspeed
from final_dataset_table)
select COUNTRY_FULL, YEAR, avg_of_country_windspeed,rnk from (select COUNTRY_FULL, YEAR, avg_of_country_windspeed,
rank() over(partition by year order by avg_of_country_windspeed desc) as rnk
from avg_windspeed_table)
where rnk=2
''')

df_Question_3.show()