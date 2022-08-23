---
title: "Data preparation"
teaching: 20
exercises: 10
questions:
- "Why are some common steps in data preparation?"
- "What is SQL and why is it often needed?"
- "What do we partition data at the start of a project?"
- "What is the purpose of setting a random state when partitioning?"
- "Should we impute missing values before or after partitioning?"
objectives:
- "Explore characteristics of our dataset."
- "Partition data into training and test sets."
- "Encode categorical values."
- "Use scaling to pre-process features."
keypoints:
- "Data pre-processing is arguably the most important task in machine learning."
- "SQL is the tool that we use to extract data from database systems."
- "Data is typically partitioned into training and test sets."
- "Setting random states helps to promote reproducibility."
---

## Sourcing and accessing data

Machine learning helps us to find patterns in data, so sourcing and understanding data is key. Unsuitable or poorly managed data will lead to a poor project outcome, regardless of the modelling approach. 

We will be using an open access subset of the [eICU Collaborative Research Database](https://eicu-crd.mit.edu/about/eicu/), a publicly available dataset comprising deidentified physiological data collected from critically ill patients. For simplicity, we will be working with a pre-prepared CSV file that comprises data extracted from a [demo version of the dataset](https://doi.org/10.13026/4mxk-na84). 

Let's begin by loading this data:

```python
import pandas as pd

# load the data
cohort = pd.read_csv('./eicu_cohort.csv')
cohort.head()
```

Learning to extract data from sources such as databases and file systems is a key skill in machine learning. Familiarity with Python and Structured Query Language (SQL) will equip you well for these tasks. For reference, the query used to extract the dataset is outlined below. Briefly, this query:

- `SELECTs` multiple columns
- `FROM` the `patient`, `apachepatientresult`, and `apacheapsvar` tables
- `WHERE` certain conditions are met.

```sql
SELECT p.gender, SAFE_CAST(p.age as int64) as age, p.admissionweight,
       a.unabridgedhosplos, a.acutephysiologyscore, a.apachescore, a.actualhospitalmortality,
       av.heartrate, av.meanbp, av.creatinine, av.temperature, av.respiratoryrate,
       av.wbc, p.admissionheight
FROM `physionet-data.eicu_crd_demo.patient` p
INNER JOIN `physionet-data.eicu_crd_demo.apachepatientresult` a
ON p.patientunitstayid = a.patientunitstayid
INNER JOIN `physionet-data.eicu_crd_demo.apacheapsvar` av
ON p.patientunitstayid = av.patientunitstayid
WHERE apacheversion LIKE 'IVa'
```

## Knowing the data

Before moving ahead on a project, it is important to understand the data. Having someone with domain knowledge - and ideally first hand knowledge of the data collection process - helps us to design a sensible task and to use data effectively.

Summarizing data is an important first step. We will want to know aspects of the data such as: extent of missingness; data types; numbers of observations. One common step is to view summary characteristics (for example, see [Table 1](https://www.nature.com/articles/s41746-018-0029-1/tables/1) of the paper by Rajkomar et al.).

Let's generate a similar table for ourselves:


```python
# !pip install tableone
from tableone import tableone

# rename columns
rename = {"unabridgedhosplos":"length of stay",
          "meanbp": "mean blood pressure",
          "wbc": "white cell count"}

# view summary characteristics
t1 = tableone(cohort, groupby="actualhospitalmortality", rename=rename)
print(t1.tabulate(tablefmt = "github"))
```

```
|                                 |         | Missing   | Overall      | ALIVE        | EXPIRED      |
|---------------------------------|---------|-----------|--------------|--------------|--------------|
| n                               |         |           | 235          | 195          | 40           |
| gender, n (%)                   | Female  | 0         | 116 (49.4)   | 101 (51.8)   | 15 (37.5)    |
|                                 | Male    |           | 118 (50.2)   | 94 (48.2)    | 24 (60.0)    |
|                                 | Unknown |           | 1 (0.4)      |              | 1 (2.5)      |
| age, mean (SD)                  |         | 9         | 61.9 (15.5)  | 60.5 (15.8)  | 69.3 (11.5)  |
| admissionweight, mean (SD)      |         | 5         | 87.6 (28.0)  | 88.6 (28.8)  | 82.3 (23.3)  |
| length of stay, mean (SD)       |         | 0         | 9.2 (8.6)    | 9.6 (7.5)    | 6.9 (12.5)   |
| acutephysiologyscore, mean (SD) |         | 0         | 59.9 (28.1)  | 54.5 (23.1)  | 86.7 (34.7)  |
| apachescore, mean (SD)          |         | 0         | 71.2 (30.3)  | 64.6 (24.5)  | 103.5 (34.9) |
| heartrate, mean (SD)            |         | 0         | 108.7 (33.1) | 107.9 (30.6) | 112.9 (43.2) |
| mean blood pressure, mean (SD)  |         | 0         | 93.2 (47.0)  | 92.1 (45.4)  | 98.6 (54.5)  |
| creatinine, mean (SD)           |         | 0         | 1.0 (1.7)    | 0.9 (1.7)    | 1.7 (1.6)    |
| temperature, mean (SD)          |         | 0         | 35.2 (6.5)   | 36.1 (3.9)   | 31.2 (12.4)  |
| respiratoryrate, mean (SD)      |         | 0         | 30.7 (15.2)  | 29.9 (15.1)  | 34.3 (15.6)  |
| white cell count, mean (SD)     |         | 0         | 10.5 (8.4)   | 10.7 (8.2)   | 9.7 (9.7)    |
| admissionheight, mean (SD)      |         | 2         | 168.0 (12.8) | 167.7 (13.4) | 169.4 (9.1)  |
```
{: .output}

> ## Exercise
> A) What is the approximate percent mortality in the eICU cohort?  
> B) Which variables appear noticeably different in the "Alive" and "Expired"  groups?  
> C) How does the in-hospital mortality differ between the eICU cohort and the ones in [Rajkomar et al](https://www.nature.com/articles/s41746-018-0029-1/tables/1)?  
> > ## Solution
> > A) Approximately 17% (40/235)   
> > B) Several variables differ, including age, length of stay, acute physiology score, heart rate, etc.  
> > A) The Rajkomar et al dataset has significantly lower in-hospital mortality (~2% vs 17%).  
> {: .solution}
{: .challenge}

## Encoding

It is often the case that our data includes categorical values. In our case, for example, the binary outcome we are trying to predict - in hospital mortality - is recorded as "ALIVE" and "EXPIRED". Some models can cope with taking this text as an input, but many cannot. We can use label encoding to convert the categorical values to numerical representations.

```python
# check current type
print(cohort['actualhospitalmortality'].dtypes)

# convert to a categorical type
cohort['actualhospitalmortality'] = cohort['actualhospitalmortality'].astype('category')

# add the encoded value to a new column
cohort['actualhospitalmortality_enc'] = cohort['actualhospitalmortality'].cat.codes
cohort[['actualhospitalmortality_enc','actualhospitalmortality']].tail(7)
```

```
   actualhospitalmortality_enc actualhospitalmortality
0                            0                   ALIVE
1                            0                   ALIVE
2                            0                   ALIVE
3                            1                 EXPIRED
4                            1                 EXPIRED
```

We'll encode the gender in the same way:

```python
# convert to a categorical type
cohort['gender'] = cohort['gender'].astype('category')
cohort['gender'] = cohort['gender'].cat.codes
```

Another popular encoding that you will come across in machine learning is "one hot encoding". In one hot encoding, categorical variables are represented as a binary column for each category. The "one hot" refers to the fact that the category can flip between "hot" (active, 1) or inactive (0). In the example above, `actualhospitalmortality` would be encoded as two columns: `ALIVE` and `EXPIRED`, each containing a list of 0s and 1s.

## Partitioning

Typically we will want to split our data into a training set and "held-out" test set. The training set is used for building our model and our test set is used for evaluation. A split of ~70% training, 30% test is common.

![Train and test set](../fig/train_test.png){: width="600px"}

To ensure reproducibility, we should set the random state of the splitting method. This means that Python's random number generator will produce the same "random" split in future.

```python
from sklearn.model_selection import train_test_split

x = cohort.drop('actualhospitalmortality', axis=1)
y = cohort['actualhospitalmortality']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 42)
```

## Missing data

Certain types of models - for example some decision trees - are able to implicitly handle missing data. For our logistic regression, we will need to impute values. We will take a simple approach of replacing with the median. 

With physiological data, imputing the median typically implies that the missing observation is not a cause for concern. In hospital you do not want to be the interesting patient!

To avoid data leaking between our training and test sets, we take the median from the training set only. The training median is then used to impute missing values in the held-out test set.

```python
# impute missing values from the training set
x_train = x_train.fillna(x_train.median())
x_test = x_test.fillna(x_train.median())
```

It is often the case that data is not missing at random. For example, the presence of blood sugar observations may indicate suspected diabetes. To use this information, we can choose to create missing data features comprising of binary "is missing" flags. 

## Normalisation

Lastly, normalisation - scaling variables so that they span consistent ranges - can be important, particularly for models that rely on distance based optimisation metrics.

As with creating train and test splits, it is a common enough task that there are plenty of pre-built functions for us to choose from. We will choose a popular scaler that transforms features in our training set to a range between zero and one.

Outliers in features can have a negative impact on the normalisation process - they can essentially squash non-outliers into a small space - so they may need special treatment.

```python
# Define the scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Alternative is zero mean, unit variance
# Subtract mean, divide by standard deviation
# from sklearn.preprocessing import StandardScaler

# fit the scaler on the training dataset
scaler.fit(x_train)

# scale the training set
x_train = scaler.transform(x_train)

# scale the test set
x_test = scaler.transform(x_test)
```

{% include links.md %}