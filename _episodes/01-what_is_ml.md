---
title: "Introduction"
teaching: 20
exercises: 10
questions:
- "What is machine learning?"
- "What is the relationship between machine learning, AI, and statistics?"

objectives:
- "Recognise what is meant by machine learning."
- "Have an appreciation of the difference between supervised and unsupervised learning."

keypoints:
- "In machine learning, models learn rules from data."
- "In supervised learning, the target in our training data is labelled."
- "Machine learning borrows heavily from fields such as statistics and computer science."

---

<!-- 
TODO: reorganise. let's make the first section about the prediction task.

# Guidelines and quality criteria for artificial intelligence-based prediction models in healthcare
https://www.nature.com/articles/s41746-021-00549-7

Reference OECD definition: https://oecd.ai/en/wonk/a-first-look-at-the-oecds-framework-for-the-classification-of-ai-systems-for-policymakers
-->

## Traditional programming

We are all familiar with the idea of applying rules to data to gain insights and make decisions. From our parents or teachers, for example, we learn that the normal body temperature is ~37 °C (~98.5 °F), and that higher or lower temperatures can be cause for concern.

As programmers we understand how to codify these rules. If we were developing software for a hospital to flag patients at risk of deterioration, we might create early-warning rules such as those below:

```python
def has_fever(temp_c)
    if temp_c > 38:
        return True
    else:
        return False
```

## Machine learning

With machine learning, we modify this approach. Instead, we give data _and_ insights to a framework that can learn and apply the rules for itself. As the volume and complexity of data increases, so does the value of having models that can generate new rules.

![Machine learning](../fig/placeholder.png)

There are ongoing and often polarised debates about the relationship between statistics, machine learning, and "A.I". There are also plenty of familiar jokes and memes like this one by [sandserifcomics](https://www.instagram.com/sandserifcomics/).

![Poorly fitted data](../fig/section1-fig2.jpeg){: width="600px"}

Keeping out of the fight, a slightly hand-wavy, non-controversial take might be:

- *Statistics*: A well-established field of mathematics concerned with methods for collecting, analyzing, interpreting and presenting empirical data.
- *Artificial intelligence*: The goal of conferring human-like intelligence to machines. "A.I." has become widely used by the media to describe any kind of sophisticated computer model, so some people have taken to using "Artificial General Intelligence" (A.G.I.) for clarity.
- *Machine learning*: A set of computational methods that learn rules from data, often with the goal of prediction. Borrows from other disciplines, notably statistics and computer science.

> ## Exercise
> A) Question
> 
> > ## Solution
> > A) Solution
> {: .solution}
{: .challenge}

## Introduction to machine learning

Over the course of four half-day lessons, we will explore key concepts in machine learning. Our focus will be on *supervised* machine learning, a category of machine learning that involves the use of labelled datasets to train models for classification and prediction. Supervised machine learning can be contrasted to *unsupervised* machine learning, which attempts to identify meaningful patterns within unlabelled datasets.

In this introductory lesson we develop and evaluate a simple predictive model, touching on some of the core concepts and techniques that we come across in machine learning projects. In later lessons, we take a deeper dive into two popular families of machine learning models - decision trees and artificial neural networks. We then finish with a focus on what it means to be a responsible practitioner of machine learning. 

{% include links.md %}