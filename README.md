# Pickling in Python

>_Well, save my model and call me pickled!_

## Why pickle?  What do I pickle?
If you have made a model and want to save it for later use, pickle it.  The `pickle` library is built into Python and uses one line of code to save your model to a separate file that can be called and used later, even in a completely separate file.

## Example:
This is an example of a simple linear regression predicting salary using years of experience.  The data came from [this Kaggle data set](https://www.kaggle.com/rohankayan/years-of-experience-and-salary-dataset).

Start by importing packages and data:

```python
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create data
d = {'years_experience': [1.1,1.3,1.5,2,2.2,2.9,3.2,3.7,3.9,4.5,4.9,
                          5.1,5.3,5.9,6,7.9,8.2,8.7,9,9.6,10.3,10.5],
     'salary': [39343,46205,37731,43525,39891,56642,64445,57189,63218,61111,
                67938,66029,83088,81363,93940,101302,113812,109431,105582,
                112635,122391,121872]}
df = pd.DataFrame(data=d)
```

Then, split the data and fit a linear regression:

```python
# Split data
X = pd.DataFrame(df.years_experience)
y = pd.DataFrame(df.salary)

# Fit regression
reg = LinearRegression().fit(X,y)
```

We can print the single coefficient and the intercept to compare with the new object when we unpickle it:

```python
print(f"Coefficient: {round(reg.coef_[0][0],2)}")  # 9267.24
print(f"Intercept: {round(reg.intercept_[0],2)}")  # 27178.6
```

Lastly, to save the regression model object (`reg`), we use the `dump` method to drop it into a file in the same directory, in this case called `save.p`.  The `wb` argument is telling the pickle library to write to a binary file.

```python
# Pickle the regression model object
pickle.dump(reg, open("save.p", "wb"))
```



## Considerations
A couple of things to keep in mind:
- When you unpickle something, you'll need to be running on the same version of Python.
- Only open pickle files over which you have had 100% control. You can pickle just about any object, including malicious code, so be cautious.
