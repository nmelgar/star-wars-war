# Star Wars Data Analysis
<hr>

For this project, your client would like to use the Star Wars survey data to figure out if they can predict an interviewing job candidate’s current income based on a few responses about Star Wars movies.

## Background
<hr>

Survey data is notoriously difficult to munge. Even when the data is recorded cleanly the options for ‘write in questions’, ‘choose from multiple answers’, ‘pick all that are right’, and ‘multiple choice questions’ makes storing the data in a tidy format difficult. </br>

In 2014, FiveThirtyEight surveyed over 1000 people to write the article titled, [America’s Favorite ‘Star Wars’ Movies (And Least Favorite Characters)](https://fivethirtyeight.com/features/americas-favorite-star-wars-movies-and-least-favorite-characters/). They have provided the data on [GitHub](https://github.com/fivethirtyeight/data/tree/master/star-wars-survey). </br>

For this project, your client would like to use the Star Wars survey data to figure out if they can predict an interviewing job candidate’s current income based on a few responses about Star Wars movies. </br>

## Data
<hr>

[StarWars.csv](https://github.com/fivethirtyeight/data/raw/master/star-wars-survey/StarWars.csv) </br>
[Information / Article](https://fivethirtyeight.com/features/americas-favorite-star-wars-movies-and-least-favorite-characters/)

## Questions and Tasks
<hr>

For Project 1 the answer to each question should include a chart and a written response. The years labels on your charts should not include a comma. At least two of your charts must include reference marks.

1. Shorten the column names and clean them up for easier use with pandas. Provide a table or list that exemplifies how you fixed the names.
2. Clean and format the data so that it can be used in a machine learning model. As you format the data, you should complete each item listed below. In your final report provide example(s) of the reformatted data with a short description of the changes made.
    + a. Filter the dataset to respondents that have seen at least one film.
    + b. Create a new column that converts the age ranges to a single number. Drop the age range categorical column.
    + c. Create a new column that converts the education groupings to a single number. Drop the school categorical column
    + d. Create a new column that converts the income ranges to a single number. Drop the income range categorical column.
    + e. Create your target (also known as “y” or “label”) column based on the new income range column.
    + f. One-hot encode all remaining categorical columns.
3. Validate that the data provided on GitHub lines up with the article by recreating 2 of the visuals from the article.
4. Build a machine learning model that predicts whether a person makes more than $50k. Describe your model and report the accuracy.

## Required Technologies
<hr>

+ At least Python 3.10.11
+ Pandas
+ Altair

## Author
<hr>

+ Nefi Melgar (mel16013@byui.edu)
