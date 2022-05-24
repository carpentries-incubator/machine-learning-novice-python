---
title: "Introduction"
teaching: 20
exercises: 10
questions:
- "What is machine learning?"
- "What is the relationship between machine learning, AI, and statistics?"
- "What is meant by supervised learning?"

objectives:
- "Recognise what is meant by machine learning."
- "Have an appreciation of the difference between supervised and unsupervised learning."

keypoints:
- "Machine learning borrows heavily from fields such as statistics and computer science."
- "In machine learning, models learn rules from data."
- "In supervised learning, the target in our training data is labelled."
- "A.I. has become a synonym for machine learning."
- "A.G.I. is the loftier goal of achieving human-like intelligence."

---

## Rule-based programming

We are all familiar with the idea of applying rules to data to gain insights and make decisions. For example, we learn that human body temperature is ~37 °C (~98.5 °F), and that higher or lower temperatures can be cause for concern.

As programmers we understand how to codify these rules. If we were developing software for a hospital to flag patients at risk of deterioration, we might create early-warning rules such as those below:

```python
def has_fever(temp_c):
    if temp_c > 38:
        return True
    else:
        return False
```

## Machine learning

With machine learning, we modify this approach. Instead, we give data _and_ insights to a framework (or "model") that can learn the rules for itself. As the volume and complexity of data increases, so does the value of having models that can generate new rules.

In a 2018 paper entitled "[Scalable and accurate deep learning with electronic health records](https://www.nature.com/articles/s41746-018-0029-1)", Rajkomar and colleagues present their work to develop a "deep learning model" (a type of machine learning model) for predicting hospital mortality. A segment of the paper's introduction is reproduced below:

> In spite of the richness and potential of available data [in healthcare], scaling the development of predictive models is difficult because, for traditional predictive modeling techniques, each outcome to be predicted requires the creation of a custom dataset with specific variables. It is widely held that 80% of the effort in an analytic model is preprocessing, merging, customizing, and cleaning datasets not analyzing them for insights. This profoundly limits the scalability of predictive models.
> 
> Another challenge is that the number of potential predictor variables in the electronic health record (EHR) may easily number in the thousands, particularly if free-text notes from doctors, nurses, and other providers are included. Traditional modeling approaches have dealt with this complexity simply by choosing a very limited number of commonly collected variables to consider.
> 
> ... Recent developments in deep learning and artificial neural networks may allow us to address many of these challenges and unlock the information in the EHR. ... These systems are known for their ability to handle large volumes of relatively messy data, including errors in labels and large numbers of input variables. A key advantage is that investigators do not generally need to specify which potential predictor variables to consider and in what combinations; instead neural networks are able to learn representations of the key factors and interactions from the data itself.

> ## Exercise
> A) What is the most time consuming aspect of developing a predictive model, according to the authors?  
> B) How have "traditional" predictive models dealt with high numbers of predictor variables, according to the authors?  
> 
> > ## Solution
> > A) 80% of effort in building models is in "preprocessing, merging, customizing, and cleaning".  
> > B) Traditional modeling approaches have dealt with complexity by choosing a very limited number of variables to consider.
> {: .solution}
{: .challenge}

These paragraphs provide an example of how machine learning can help us with tasks like prediction. They also touch on an area where machine learning projects often comes under criticism. It is easy to throw tools at a problem without sufficient thought!

## Statistics, machine learning, and "AI"

There are ongoing and often polarised debates about the relationship between statistics, machine learning, and "A.I". There are also plenty of familiar jokes and memes like this one by [sandserifcomics](https://www.instagram.com/sandserifcomics/).

![Poorly fitted data](../fig/section1-fig2.jpeg){: width="600px"}

Keeping out of the fight, a slightly hand-wavy, non-controversial take might be:

- *Statistics*: A well-established field of mathematics concerned with methods for collecting, analyzing, interpreting and presenting empirical data.
- *Machine learning*: A set of computational methods that learn rules from data, often with the goal of prediction. Borrows from other disciplines, notably statistics and computer science.
- *Deep learning*: A subfield of machine learning that focuses on more complex "artificial neural network" algorithms.
- *Artificial intelligence*: The goal of conferring human-like intelligence to machines. "A.I." has become popularly used as a synonym for machine learning, so researchers working on the goal of intelligent machines have taken to using "Artificial General Intelligence" (A.G.I.) for clarity.

## Supervised vs unsupervised learning

Over the course of four half-day lessons, we will explore key concepts in machine learning. In this introductory lesson we develop and evaluate a simple predictive model, touching on some of the core concepts and techniques that we come across in machine learning projects. Our goal will be to become familiar with the kind of language used in papers such as "[Scalable and accurate deep learning with electronic health records](https://www.nature.com/articles/s41746-018-0029-1)".

Our focus will be on *supervised* machine learning, a category of machine learning that involves the use of labelled datasets to train models for classification and prediction. Supervised machine learning can be contrasted to *unsupervised* machine learning, which attempts to identify meaningful patterns within unlabelled datasets.

In later lessons, we take a deeper dive into two popular families of machine learning models - decision trees and artificial neural networks. We then explore what it means to be a responsible practitioner of machine learning. 

> ## Exercise
> A) We have laboratory test data on patients admitted to a critical care unit and we are trying to identify patients with an emerging, rare disease. There are no labels to indicate which patients have the disease, but we believe that the infected patients will have very distinct characteristics. Do we look for a supervised or unsupervised machine learning approach?   
> B) We would like to predict whether or not patients will respond to a new drug that is under development based on several genetic markers. We have a large corpus of clinical trial data that includes both genetic markers of patients and their response the new drug. Do we use a supervised or unsupervised approach?
> 
> > ## Solution
> > A) The prediction targets are not labelled, so an unsupervised learning approach would be appropriate. Our hope is that we will see a unique cluster in the data that pertains to the emerging disease.  
> > B) We have both genetic markers and known outcomes, so in this case supervised learning is appropriate.
> {: .solution}
{: .challenge}

{% include links.md %}