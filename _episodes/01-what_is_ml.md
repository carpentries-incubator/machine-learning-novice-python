---
title: "Machine learning"
teaching: 20
exercises: 10
questions:
- "What is machine learning?"

objectives:
- "Recognise what is meant by machine learning."

keypoints:
- "In machine learning, models learn rules from data."

---

{% include links.md %}

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

## Machine learning in the wild

Machine learning has been creeping into our everyday lives for some time, in areas including:

- Advertising
- National security
- Job recruitment
- Criminal justice

This rise is rapidly expanding the demand for our personal data. Technology for harvesting and analysing this data is rapidly advancing, and it is fair to say that governance is playing catch up. 

> ## Exercise
> A) Question
> 
> > ## Solution
> > A) Solution
> {: .solution}
{: .challenge}
