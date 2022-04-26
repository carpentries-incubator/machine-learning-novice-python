---
layout: lesson
root: .  # Is the only page that doesn't follow the pattern /:path/index.html
permalink: index.html  # Is the only page that doesn't follow the pattern /:path/index.html
---

This lesson provides an introduction to some of the common methods and terminologies used in machine learning research. We cover areas such as data preparation and resampling, model building, and model evaluation. 

It is a prerequisite for the other lessons in the machine learning curriculum. In later lessons we explore tree-based models for prediction, neural networks for image classification, and responsible machine learning. 

## Scenario: Prediction of patient outcomes in the intensive care

Intensive care units are home to sophisticated monitoring systems, helping carers to support the lives of the sickest patients within a hospital. These monitoring systems produce large volumes of data that could be used to improve patient care.

One potential use of the data is to predict the outcome of patients using information available on the day of admission. The predictions may be helpful for resource planning and to assist with family discussions. Prediction is a task that is well-suited to machine learning. 

We will be using a set of data that has been extracted from the eICU Collaborative Research Database, a publicly available dataset comprising deidentified physiological data collected from critically ill patients.

<!-- this is an html comment -->

{% comment %} This is a comment in Liquid {% endcomment %}

> ## Prerequisites
>
> You need to understand the basics of Python before tackling this lesson. The lesson sometimes references Jupyter Notebook although you can use any Python interpreter mentioned in the [Setup][lesson-setup].
{: .prereq}

### Getting Started

To get started, follow the directions on the "[Setup][lesson-setup]" page to download data and install a Python interpreter.

{% include links.md %}
