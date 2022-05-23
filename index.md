---
layout: lesson
root: .  # Is the only page that doesn't follow the pattern /:path/index.html
permalink: index.html  # Is the only page that doesn't follow the pattern /:path/index.html
---

This lesson provides an introduction to some of the common methods and terminologies used in machine learning research. We cover areas such as data preparation and resampling, model building, and model evaluation. 

It is a prerequisite for the other lessons in the machine learning curriculum. In later lessons we explore tree-based models for prediction, neural networks for image classification, and responsible machine learning. 

## Predicting the outcome of critical care patients

Critical care units are home to sophisticated monitoring systems, helping carers to support the lives of the sickest patients within a hospital. These monitoring systems produce large volumes of data that could be used to improve patient care.

![Patient in the ICU](../fig/icu_patient.png){: width="600px"}

There is benefit in being able to predict the outcome of patients, for example for resource planning and to assist with family discussions. This is a task that is well-suited to machine learning.

We will develop a model for predicting the outcome of critical care patients using physiological data available on the first day of admission to the intensive care unit.

The dataset used in this lesson was extracted from the [eICU Collaborative Research Database](http://dx.doi.org/10.1038/sdata.2018.178), a publicly available dataset comprising deidentified physiological data collected from critically ill patients.

<!-- this is an html comment -->

{% comment %} This is a comment in Liquid {% endcomment %}

> ## Prerequisites
>
> You need to understand the basics of Python before tackling this lesson. The lesson sometimes references Jupyter Notebook although you can use any Python interpreter mentioned in the [Setup][lesson-setup].
{: .prereq}

### Getting Started

To get started, follow the directions on the "[Setup][lesson-setup]" page to download data and install a Python interpreter.

{% include links.md %}
