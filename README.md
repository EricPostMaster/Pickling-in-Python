# Pickling in Python

>_Well, save my model and call me pickled!_ :cucumber:

## Why pickle?  What do I pickle?
Pickling does exactly what it sounds like: it preserves something for later.*  If you train and score a model and want to save it for later or deploy it for use on new data, you can pickle it so you don't have to retrain the model every time you want to use it.  The `pickle` module is built into Python and uses one line of code to save your model to a separate file that can be called and used later, even in a completely separate script or notebook.

<sub>* Pickling a process called "serialization," which basically means it breaks your object down into a single-file stream of bytes and saves them in order.  In this tutorial, the object is a LinearRegression sklearn model.</sub>

## Example:
This is an example of a simple linear regression predicting salary using years of experience.  The data came from [this Kaggle data set](https://www.kaggle.com/rohankayan/years-of-experience-and-salary-dataset).

### Create & pickle a model
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

Then, separate the predictor and target and fit a linear regression:

```python
# Split data
X = pd.DataFrame(df.years_experience)
y = pd.DataFrame(df.salary)

# Fit regression
reg = LinearRegression().fit(X,y)  #<-- We're going to pickle this in a minute
```

We can print the single coefficient and the intercept to compare with the new object when we unpickle it:

```python
print(f"Coefficient: {round(reg.coef_[0][0],2)}")  # 9267.24
print(f"Intercept: {round(reg.intercept_[0],2)}")  # 27178.6
```

Lastly, to save the regression model object (`reg`), we use the `.dump()` method to save the model in a file called `save.p` in the working directory.  The `wb` argument is telling the pickle module to write (i.e., create) a new file.

I like to think of it like storing something in a box. We are going to `open` a box and `dump` our `reg` model into it.  Then we'll label it `pickled_model.p` so we can find it easily later.

```python
# Pickle the regression model object
with open("pickled_model.p", "wb") as p:
    pickle.dump(reg, p)
```

### Unpickle the model
Now, we will create a totally new file and unpickle the model we just created.

Start by importing packages and creating some new data:
```python
import pickle
import pandas as pd

# New data
d = {'years_experience': [3,3.2,4,4,4.1,6.8,7.1,9.5],
     'salary': [60150,54445,55794,56957,57081,91738,98273,116969]}
test = pd.DataFrame(data=d)

# Separate data into X and y
X_test = pd.DataFrame(test.years_experience)
y_test = pd.DataFrame(test.salary)
```

Now, load the pickled model using the `.load()` method, opening the file we saved earlier (`pickled_model.p`) and reading it with the `rb` argument.

```python
# Unpickle the regression model object
with open("pickled_model.p", "rb") as p:
    new_reg = pickle.load(p)
```

We can print the coefficient and intercept to verify that it's the same model:

```python
print(f"Coefficient: {round(new_reg.coef_[0][0],2)}")  # 9267.24
print(f"Intercept: {round(new_reg.intercept_[0],2)}")  # 27178.6
```

They're the same! :tada:

Now we can make predictions on new data and evaluate model quality with R<sup>2</sup> and the Mean Absolute Percentage Error (MAPE):

```python
# R-Squared
r_squared = new_reg.score(X_test, y_test)
print(f"R-Squared: {round(r_squared*100, 2)}")  # 93.95

# Predictions
preds = new_reg.predict(X_test)

# MAPE
mape = abs((y_test - preds)/y_test).mean()
print(f"MAPE: {round(mape[0]*100, 2)}%")  # 7.96%
```

The actual results in this example aren't that important, but I wanted to go through the whole example.  I also wanted to demonstrate that you don't have to import the LinearRegression module when you unpickle the file because the methods required for predicting and scoring were pickled in the original file.  Nifty! :smiley:


## Considerations & Conclusion
A couple of things to keep in mind:
- When you unpickle something, you'll need to be running on the same version of Python.
- Only unpickle files that you trust completely.  You can pickle just about any object, including malicious code, so be extra cautious.
- The `pickle` module is exclusive to Python.  If you are planning to use your object in a different language or want the pickled object to be readable by humans, consider JSON serialization instead.  Wouldn't it be cool if there was a library that could pickle to JSON??  Oh look, it's [jsonpickle](https://github.com/jsonpickle/jsonpickle)!


Now, go forth and pickle!


## Other Resources
[Official pickle module docs](https://docs.python.org/3/library/pickle.html)  
[Save and Load Machine Learning Models in Python with scikit-learn](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)


## Thanks
Thanks to [Mark Freeman II](https://www.linkedin.com/in/mafreeman2) and [Timo Voipio](https://www.linkedin.com/in/t-voipio) for their suggestions on improving this repo :+1:
