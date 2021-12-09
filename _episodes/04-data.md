---
title: "Data preparation"
teaching: 20
exercises: 10
questions:
- "What are training and test sets?"
objectives:
- "Apply key steps in data pre-processing."
keypoints:
- "Data pre-processing is as important as modelling."
---

## Mortality prediction

Machine learning helps us to find patterns in data, so sourcing and pre-processing the right data is key. Unsuitable or poorly managed data will lead to a poor project outcome, regardless of the modelling approach.

For the rest of this lesson, we will follow a simple workflow to develop a logistic regression model for predicting the outcome of critical care patients using physiological data available on the first day of admission.

Data preparation is often the most time consuming aspect of a machine learning project. In this section, we will touch on some common themes in data preparation.

## Sourcing and accessing data

Sourcing and accessing data for a project may be challenging, especially in cases such as health where patient privacy must be respected. Typically a sensitive dataset will need to be "deidentified" before it is available for analysis.

Deidentification is the process of removing identifiers such as names, data, ages, and other personal information to maintain privacy. Deidentification is not a well-defined term and its definition varies widely by location and institution.

For our mortality prediction project, we will be using an open access subset of the eICU Collaborative Research Database, a publicly available dataset comprising deidentified physiological data collected from critically ill patients.

The majority of observations are variables underlying the APACHE predictions. Acute Physiology Age Chronic Health Evaluation (APACHE) consists of a groups of equations used for predicting outcomes in critically ill patients. See: http://eicu-crd.mit.edu/eicutables/apachePredVar/

## Data extraction and integration

You will need to be able to extract data from databases and to parse data storage formats (e.g. CSV, JSON). Familiarity with Python and Structured Query Language (SQL) will equip you well for these tasks.

In cases where data is sourced from multiple systems or locations, data harmonisation may need to be considered. For example, two different hospital systems may have different data structures and terminologies. Important, ongoing efforts to create common data models and terminologies seek to help.

For simplicity, we will be working with a pre-prepared CSV file that comprises data extracted from the eICU Collaborative Research Database (Demo). We will not discuss the query in detail here, but the main points to note are that we:
- extract multiple columns from the data with `SELECT`
- join data across the `patient`, `apachepatientresult`, and `apacheapsvar` tables
- restrict our results to a set of conditions.

```sql
SELECT p.unitadmitsource, p.gender, SAFE_CAST(p.age as int64) as age, p.admissionweight,
       a.actualiculos, a.unabridgedhosplos, p.unittype, p.unitstaytype, a.acutephysiologyscore,
       p.dischargeweight, p.admissionheight, a.apachescore, a.actualhospitalmortality,
       av.heartrate, av.meanbp, av.creatinine, av.bun, av.temperature, av.respiratoryrate,
       av.pco2, av.pao2, av.ph, av.wbc, av.vent, av.hematocrit, a.unabridgedunitlos, 
       a.predictedicumortality, a.apachescore, a.actualicumortality, a.predictediculos
FROM `physionet-data.eicu_crd_demo.patient` p
INNER JOIN `physionet-data.eicu_crd_demo.apachepatientresult` a
ON p.patientunitstayid = a.patientunitstayid
INNER JOIN `physionet-data.eicu_crd_demo.apacheapsvar` av
ON p.patientunitstayid = av.patientunitstayid
WHERE apacheversion LIKE 'IVa'
```

## Knowing your data

Before moving ahead on a project, it is important to understand the data that we are working with. Having someone with domain knowledge - and ideally first hand knowledge of the data collection process - helps us to design a sensible task and to use data effectively.

Summarizing data is an important first step. We will want to know aspects of the data such as: extent of missingness; data types; numbers of observations. One common step is to view summary characteristics.

```python
# import libraries
import pandas as pd
from tableone import tableone

# load the data
cohort = pd.read_csv('./data/eicu_cohort_ph.csv')
cohort.describe()
```

```
                      count        mean        std    min     25%     50%      75%     max
age                   111.0   60.432432  14.516211  22.00  52.500   60.00   69.000   88.00
admissionweight       113.0   87.519027  30.991813  45.50  67.400   81.60  101.700  262.40
acutephysiologyscore  117.0   51.982906  22.936939  -1.00  37.000   49.00   62.000  140.00
apachescore           117.0   63.119658  24.935490  -1.00  46.000   61.00   76.000  158.00
ph                    117.0    7.354701   0.089970   7.05   7.304    7.36    7.417    7.56
pco2                  117.0   42.852137  12.745172  26.00  35.000   40.30   47.900   94.00
respiratoryrate       117.0   29.692308  14.925945  -1.00  24.000   33.00   39.000   60.00
wbc                   117.0   10.291709   8.397978  -1.00   5.600    9.70   15.500   32.60
creatinine            117.0    0.892120   1.591968  -1.00   0.370    0.88    1.380    9.12
bun                   117.0   20.606838  24.195667  -1.00  -1.000   15.00   28.000  123.00
heartrate             117.0  105.111111  31.009114  23.00  93.000  110.00  124.000  176.00
intubated             117.0    0.290598   0.455991   0.00   0.000    0.00    1.000    1.00
vent                  117.0    0.000000   0.000000   0.00   0.000    0.00    0.000    0.00
temperature           117.0   35.810256   4.941121  -1.00  36.100   36.40   36.720   40.20
```
{: .output}

Visualising data is also key to understanding it.

```python
cohort.age.hist(bins=20)
plt.show()

cohort.unittype.value_counts().plot(kind='barh')
plt.show()
```


```python
# view summary characteristics
t1 = tableone(cohort, groupby="actualhospitalmortality")
print(t1.tabulate(tablefmt = "github"))
```

```
|                                        |                      | Missing   | Overall      | ALIVE        | EXPIRED      |
|----------------------------------------|----------------------|-----------|--------------|--------------|--------------|
| n                                      |                      |           | 117          | 98           | 19           |
| unitadmitsource, n (%)                 | Direct Admit         | 1         | 12 (10.3)    | 12 (12.4)    |              |
|                                        | Emergency Department |           | 53 (45.7)    | 41 (42.3)    | 12 (63.2)    |
|                                        | Floor                |           | 19 (16.4)    | 13 (13.4)    | 6 (31.6)     |
|                                        | Operating Room       |           | 28 (24.1)    | 28 (28.9)    |              |
|                                        | Recovery Room        |           | 2 (1.7)      | 2 (2.1)      |              |
|                                        | Step-Down Unit (SDU) |           | 2 (1.7)      | 1 (1.0)      | 1 (5.3)      |
| gender, n (%)                          | Female               | 0         | 60 (51.3)    | 51 (52.0)    | 9 (47.4)     |
|                                        | Male                 |           | 57 (48.7)    | 47 (48.0)    | 10 (52.6)    |
| age, mean (SD)                         |                      | 6         | 60.4 (14.5)  | 59.3 (14.4)  | 66.3 (13.8)  |
| admissionweight, mean (SD)             |                      | 4         | 87.5 (31.0)  | 90.3 (32.4)  | 72.9 (16.2)  |
| unittype, n (%)                        | CCU-CTICU            | 0         | 12 (10.3)    | 12 (12.2)    |              |
|                                        | CSICU                |           | 3 (2.6)      | 3 (3.1)      |              |
|                                        | CTICU                |           | 10 (8.5)     | 10 (10.2)    |              |
|                                        | Cardiac ICU          |           | 12 (10.3)    | 4 (4.1)      | 8 (42.1)     |
|                                        | MICU                 |           | 7 (6.0)      | 6 (6.1)      | 1 (5.3)      |
|                                        | Med-Surg ICU         |           | 68 (58.1)    | 58 (59.2)    | 10 (52.6)    |
|                                        | Neuro ICU            |           | 3 (2.6)      | 3 (3.1)      |              |
|                                        | SICU                 |           | 2 (1.7)      | 2 (2.0)      |              |
| unitstaytype, n (%)                    | admit                | 0         | 111 (94.9)   | 93 (94.9)    | 18 (94.7)    |
|                                        | readmit              |           | 4 (3.4)      | 3 (3.1)      | 1 (5.3)      |
|                                        | transfer             |           | 2 (1.7)      | 2 (2.0)      |              |
| acutephysiologyscore, mean (SD)        |                      | 0         | 52.0 (22.9)  | 47.4 (17.9)  | 75.7 (30.9)  |
| apachescore, mean (SD)                 |                      | 0         | 63.1 (24.9)  | 57.4 (19.2)  | 92.4 (30.6)  |
| ph, mean (SD)                          |                      | 0         | 7.4 (0.1)    | 7.4 (0.1)    | 7.3 (0.1)    |
| pco2, mean (SD)                        |                      | 0         | 42.9 (12.7)  | 43.2 (12.2)  | 41.1 (15.4)  |
| respiratoryrate, mean (SD)             |                      | 0         | 29.7 (14.9)  | 29.1 (14.8)  | 33.0 (15.3)  |
| wbc, mean (SD)                         |                      | 0         | 10.3 (8.4)   | 10.7 (8.0)   | 8.1 (10.3)   |
| creatinine, mean (SD)                  |                      | 0         | 0.9 (1.6)    | 0.8 (1.5)    | 1.5 (2.0)    |
| bun, mean (SD)                         |                      | 0         | 20.6 (24.2)  | 18.5 (23.6)  | 31.7 (24.6)  |
| heartrate, mean (SD)                   |                      | 0         | 105.1 (31.0) | 103.2 (28.9) | 115.0 (39.6) |
| intubated, mean (SD)                   |                      | 0         | 0.3 (0.5)    | 0.3 (0.5)    | 0.3 (0.5)    |
| vent, mean (SD)                        |                      | 0         | 0.0 (0.0)    | 0.0 (0.0)    | 0.0 (0.0)    |
| temperature, mean (SD)                 |                      | 0         | 35.8 (4.9)   | 36.2 (3.9)   | 34.0 (8.5)   |
| actualhospitalmortality_enc, mean (SD) |                      | 0         | 0.2 (0.4)    | 0.0 (0.0)    | 1.0 (0.0)    |
```
{: .output}

## Partitioning

Typically we will want to split our data into a training set and "held-out" test set. The training set is used for building our model and our test set is used for evaluation. A split of ~70% training, 30% test is common.

To ensure reproducibility, we should set the random state of the splitting method. This means that Python's random number generator will produce the same "random" split in future.

```python
from sklearn.model_selection import train_test_split

x = cohort.drop('actualhospitalmortality', axis=1)
y = cohort['actualhospitalmortality']
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7,
                                                    random_state =  42)
```

## Data leaks

Any data preparation prior to fitting the model should be carried out before partitioning. This helps us to avoid "data leakage" where knowledge of the test dataset is used to improve the model. 

For example, if done prior to partitioning, both of the following steps could leak information about our test set into our training set:

- Filling missing data.
- Scaling the range of a variable.

Care is also needed when deciding on the partitioning approach. It is often enough to split the data randomly, but this is not always a good approach. When splitting, consider whether there is potential for information to "leak" into our test set.

Data leakage can invalidate our results, for example by giving us an overly optimistic estimates of model performance.

## Cleaning

Usually while reviewing our data, we notice issues that would benefit from dealing with before training our model. In health data it is common to see issues such as:

- Temperatures have been incorrectly recorded in Fahrenheit, rather than Centigrade. We may want to apply a rule to transform temperatures in Fahrenheit to Centigrade 
- Non-standardised recording of concepts or terms (for example, heart rate appearing as both "HR" and "heart rate").

```python
[TODO: add cleaning step]
```

## Encoding

It is often the case that our data includes categorical values. In our case, for example, the binary outcome we are trying to predict - in hospital mortality - is recorded as "ALIVE" and "EXPIRED". Some models can cope with taking this text as an input, but many cannot. 

For our logistic regression model, we will need to encode the categorical values as numerical values. We will encode these labels by converting them to numbers. In our case, 0 for "ALIVE" and 1 for "EXPIRED". This is usually referred to as "label encoding".

```python
# check current type
print(cohort['actualhospitalmortality'].dtypes)

# convert to a categorical type
cohort['actualhospitalmortality'] = cohort['actualhospitalmortality'].astype('category')

# add the encoded value to a new column
cohort['actualhospitalmortality_enc'] = cohort['actualhospitalmortality'].cat.codes
cohort[['actualhospitalmortality_enc','actualhospitalmortality']].head()
```

```
   actualhospitalmortality_enc actualhospitalmortality
0                            0                   ALIVE
1                            0                   ALIVE
2                            0                   ALIVE
3                            1                 EXPIRED
4                            0                   ALIVE
```

An alternative method, often used when there is no intrinsic ordering in our variables, is "one hot encoding". 

In one hot encoding, we create a new column for each categorical value and then assign a 0 or 1 (False or True) to the column (i.e. for a single categorical variable, there is a single "hot" - or true - state).

## Missing data

Some types of models - for example some decision trees - are able to implicitly handle missing data. For our logistic regression, we will need to impute values. We will take a simple approach of replacing with the median. 

With physiological data, imputing from the median typically implies that the missing observation is a "healthy" normal. As the clinicians say, in hospital you do not want to be the interesting patient!

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
# define the scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# fit the scaler on the training dataset
scaler.fit(X_train)

# scale the training set
X_train = scaler.transform(X_train)

# scale the test set
X_test = scaler.transform(X_test)
```

{% include links.md %}