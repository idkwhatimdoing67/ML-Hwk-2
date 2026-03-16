NN.py was written and ran in VS-Code, the git repo will also be linked if commit history is desired.

Running the code itself, the command I used was "python3 NN.py"
This should provide an output of the table and then 3 charts.

The charts that this specific will be provided in the report.

Note:
The charts will appear 1 at a time and will likely have to be closed before the next one appears or the program will continue running.

With regard to the dataset used:
The link will be provided in the report but it was named "Student Performance"

When downloading this you might unzip the data and find two files, I used "student-mat.csv"

This code is also designed to take csvs separated by ; not , (as that was the format of student-mat.csv)


Summary of work:

Preprocess:
This section was designed to be identify what type of scaling to use for the data depending on a few factors

First we check column by column if it's numeric or not
    If numeric we test IQR, and p value to make final determinations on what scaler to use.

If col is categorical
    Check how many categories, if 2 we use label encoder
    If more we use one-hot encoding

Preprocess was a bit experimental to me, I've never tried out these different processes before but I wanted to give it a shot.


Train/Eval:
This section was more straightforward (at least in direction)

I used sklearn's MLPClassifier for this

First a loop structure was set up to iterate through all hyperparameters
After this I set up the MLPClassifier initialization to create a model to each parameter we iterated through

Once done, I call test methods for the varying outputs we need and set them up with a plot to observe.