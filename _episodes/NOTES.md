INtro

we will develop a model for predicting the outcome of critical care patients using physiological data available on the first day of admission to the intensive care unit.

Over the course of the lesson we will touch on common terminology used in machine learning studies. Later

- get people into ML mindset
- introduce popular concepts in ML, rather than focusing on sophisticated models
- later workshops will introduce fancy models
- 

===

> ## Exercise
> A) 
> B) 
> 
> > ## Solution
> > A) 
> > B) 
> {: .solution}
{: .challenge}

===

TP: look at this https://python-course.eu/machine-learning/. Intro to ML course.


===

HERE IS PYTHON NOVICE: https://swcarpentry.github.io/python-novice-inflammation/

===

Machine learning can be roughly separated into three categories:

Supervised learning
The machine learning program is both given the input data and the corresponding labelling. This means that the learn data has to be labelled by a human being beforehand.
Unsupervised learning
No labels are provided to the learning algorithm. The algorithm has to figure out the a clustering of the input data.
Reinforcement learning
A computer program dynamically interacts with its environment. This means that the program receives positive and/or negative feedback to improve it performance.



===

FOR GRADIENT DESCENT, USE THIS!!!!!

from sklearn.datasets import make_regression


https://gist.github.com/felipessalvatore/c2e1c09dfcb8710b847e2457620f8204#file-gradient_descent-py-L98

(REPLACE EXISTING CODE)

===

https://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines

===

https://ruder.io/optimizing-gradient-descent/

===

https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/

===



Starting with some guesswork, let's pick values for $$\beta_0$$ and $$\beta_1$$ and calculate the error. 

```python
# `mean_squared_error` computes MSE from two lists (y_true and y_hat)
from sklearn.metrics import mean_squared_error

# Define our model
def model(b0, b1, X):
    """
    Linear regression model: f(x) = bo + b1 * x. Takes array of x-values and
    outputs corresponding y-values.
    """
    return b0 + (b1 * X)

# Select coefficients
b0 = 400
b1 = -80

# Get x, y_true, and y_pred
x = cohort.ph.values
y_true = cohort.pco2.values
y_pred = model(b0, b1, x)

# Compute MSE
print(f'MSE of model: {mean_squared_error(y_true, y_pred)}')
```

How did we do? To improve our model, we need to find better values for $$\beta_0$$ and $$\beta_1$$. 

===

You will have seen linear regression models described by the following formula:

$$
E(y) = \beta_0 + \beta_1 x_1
$$

The outcome variable (length of stay) is denoted by $y$ and the explanatory variable (severity score) is denoted by $x1$. Simple linear regression models the $expectation$ of $y$, i.e. $E(y)$. $\beta_0$ is the y-axis intercept; $\beta_1$ is the slope.  

Our goal is to model the relationship between severity of illness and length, finding values for the parameters ($\beta_0$ and $\beta_1$) such that our  estimates for $y$ are as close to the truth as possible.


===

# LOOK AT MODEL SURFACE

Let's plot the mean squared error across a matrix of different values of our two parameters, $$\beta_0$$ and $$\beta_1$$, to get a better idea about what the optimal model might be.

[TODO: clean and perhaps vectorize and normalize.]

```python
import numpy as np
# library for 3d surface plot
from mpl_toolkits import mplot3d

# range of b0 and b1 values to iterate over
b0_vec = np.linspace(540, 560, num=50)
b1_vec = np.linspace(-60, -80, num=50)
xx,yy = np.meshgrid(b0_vec, b1_vec)

# create a matrix of our b0 and b1 vectors
def loss(b0, b1):
    """
    b0 and b1 can be lists.
    """
    y_true = cohort.pco2.values
    y_pred = model(b0, b1, cohort.ph.values)
    # Compute MSE
    return mean_squared_error(y_true, y_pred)

# compute MSE for each (b0, b1) point on grid
zz = np.empty([len(b0_vec),len(b1_vec)])
for n,x in enumerate(b0_vec):
    for m,y in enumerate(b1_vec):
        zz[n,m] = loss(x, y)

# plot
ax = plt.axes(projection='3d')
ax.set_xlabel('Intercept (b0)')
ax.set_ylabel('Slope (b1)')
ax.set_zlabel('Loss', rotation="vertical")
ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='plasma')
ax.set_title('Model error over different values of b0 and b1');
plt.show()
```

![Loss function](../fig/section2-fig4.png){: width="600px"}

Each (x,y) point represents a fitted line. The z-axis show the corresponding error value. Our goal is to find the values for $$\beta_0$$ and $$\beta_1$$ that correspond to the minimum loss. In the next section, we'll look at how we can search this space to find our optimal parameters for $$\beta_0$$ and $$\beta_1$$ .


====


## Gradient descent in practice

First, we initialize $$\beta_0$$ and $$\beta_1$$ with random values. For example, $$\beta_0$$ = 1 and $$\beta_1$$ = 2. Our regression formula is now:

$$
f(x) = 1 + 2 \times x
$$

Second, we calculate the partial derivative of our loss function (the MSE) with respect to $$\beta_0$$ and $$\beta_1$$. The formula for the derivatives are:

$$
\frac{\delta mse}{\delta\beta_0} = \frac{2}{n}\sum_{i=1}^{n}-(y_{i} - \hat{y}_{i})^{2}
$$

$$
\frac{\delta{mse}}{\delta\beta_1} = \frac{2}{n}\sum_{i=1}^{n}-{x}_{i}(y_{i} - \hat{y}_{i})^{2}
$$

Third, we plug these values into the following formula to find our improved parameters. $$L$$ is the learning rate, which is the size of our steps. We'll briefly discuss how we select L later.

$$
\beta_0 = \beta_0 - (L \times \delta\beta_0)
$$

$$
\beta_1 = \beta_1 - (L \times \delta\beta_1)
$$

Repeat! When successive iterations cease to decrease the loss function (or only result in very small changes), we have reached the local minima. Our model has "converged".


===

To help visualise our search task, let's begin by plotting the space that we are searching over. We'll create a matrix of different values for the weights and bias, and then we'll calculate the loss at each of these points.

```python
import numpy as np
# library for 3d surface plot
from mpl_toolkits import mplot3d

# range of b0 and b1 values to iterate over
weight_space = np.linspace(-100, 100, num=200)
bias_space = np.linspace(-100, 100, num=200)
xx,yy = np.meshgrid(weight_space, bias_space)

# prediction targets
x_true = cohort.gossis.values
y_true = cohort.hospital_los.values

# predictive model (from previous section)
def model(weight, X, bias):
    """
    Linear regression model: y_hat = wX + b. Takes array of x-values and
    outputs corresponding y-values.
    """
    return np.dot(weight, X) + bias

# loss function (from previous section)
def loss(y, y_hat):
    """
    Loss function (mean squared error)
    """
    return np.mean((y - y_hat)**2)

# compute MSE for each (b0, b1) point on grid
zz = np.empty([len(weight_space), len(bias_space)])
for n, weight in enumerate(weight_space):
    for m, bias in enumerate(bias_space):
        y_hat = model(weight, x_true, bias)
        zz[n, m] = loss(y_true, y_hat)

# plot
ax = plt.axes(projection='3d')
ax.set_xlabel('Slope (weight)')
ax.set_ylabel('Intercept (bias)')
ax.set_zlabel('Loss', rotation="vertical")
ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='plasma')
ax.set_title('Loss for different parameter values');
plt.show()
```

![Loss function](../fig/section2-fig4.png){: width="600px"}


====


We can begin by initialising our weights and bias with values of 0:

```python
n_features = 1
weights = np.zeros((n_features, 1))
bias = 0
```

Now we need to update our weights and bias to reduce the loss. To do this, we use the update rule for gradient descent. We'll skip the math for now. The code is:

```python
weights -= lr*dw
bias -= lr*db
```


=====


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

====


An alternative method, often used when there is no intrinsic ordering in our variables, is "one hot encoding". 

In one hot encoding, we create a new column for each categorical value and then assign a 0 or 1 (False or True) to the column (i.e. for a single categorical variable, there is a single "hot" - or true - state).

===


## Cleaning

Usually while reviewing our data, we notice issues that would benefit from dealing with before training our model. In health data it is common to see issues such as:

- Temperatures have been incorrectly recorded in Fahrenheit, rather than Centigrade. We may want to apply a rule to transform temperatures in Fahrenheit to Centigrade 
- Non-standardised recording of concepts or terms (for example, heart rate appearing as both "HR" and "heart rate").

```python
[TODO: add cleaning step]
```

====

$$
f(x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_n x_n
$$

where $$f(x)$$ is our outcome, $$\beta_0$$, $$\beta_0$$, $$\beta_0$$ are our model parameters, and $$x_1$$, $$x_2$$, $$x_n$$ are our explanatory variables (or, in machine learning, "features").

====

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

====


$$
\begin{bmatrix}
  x_{1}^1 & x_{2}^1 & \ldots & x_{n}^1 \\ 
  x_{1}^2 & x_{2}^2 & \ldots & x_{n}^1 \\
  \vdots & \vdots & \ldots & x_{n}^1 \\
  x_{m}^m & x_{2}^m & \ldots & x_{n}^1 \\
\end{bmatrix}
$$

===

