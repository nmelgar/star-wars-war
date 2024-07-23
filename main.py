# %%
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# %%
url = "https://github.com/fivethirtyeight/data/blob/master/star-wars-survey/StarWars.csv?raw=true"
data_sw = pd.read_csv(url, encoding="ISO-8859-1")
data_sw = data_sw.drop(index=0)

## GRAND QUESTION 1
# Shorten the column names and clean them up for easier use with pandas. Provide a table
# or list that exemplifies how you fixed the names.
# %%
# create a df to display the names of the columns olds and news
new_columns_df = pd.DataFrame(columns=["old_columns", "new_columns"])
old_cols = data_sw.columns.tolist()
# insert old column names from data_sw to new 'old_columns' column in new cols df
new_columns_df["old_columns"] = old_cols

def new_columns(df_frame):
    # new names for the columns
    df_frame.columns = [
    "respondent_id",
    "seen_any_sw_film",
    "sw_fan",
    "seen_ep1",
    "seen_ep2",
    "seen_ep3",
    "seen_ep4",
    "seen_ep5",
    "seen_ep6",
    "rank_ep1",
    "rank_ep2",
    "rank_ep3",
    "rank_ep4",
    "rank_ep5",
    "rank_ep6",
    "fav_han",
    "fav_luke",
    "fav_leia",
    "fav_anakin",
    "fav_obi",
    "fav_emperor",
    "fav_darth",
    "fav_lando",
    "fav_boba",
    "fav_c3po",
    "fav_r2d2",
    "fav_jarjar",
    "fav_padme",
    "fav_yoda",
    "shot_first",
    "expanded_universe_fam",
    "expanded_universe_fan",
    "star_trek_fan",
    "gender",
    "age",
    "income",
    "education",
    "location",
]
new_columns(data_sw)
# insert new column names in the new cols df
new_columns_df["new_columns"] = data_sw.columns.tolist()

new_columns_df.head(10)

## GRAND QUESTION 2
# Clean and format the data so that it can be used in a machine learning model.
# As you format the data, you should complete each item listed below.
# In your final report provide example(s) of the reformatted data with a
# short description of the changes made.

# %%
# add respondent id an easier to read number
data_sw["respondent_id"] = range(1, len(data_sw) + 1)

# %%
#     Filter the dataset to respondents that have seen at least one film
data_sw = data_sw[data_sw["seen_any_sw_film"] == "Yes"]

# %%
# fill NaN for numerical columns usign the median
num_cols = data_sw.select_dtypes(include=["float64", "int64"]).columns
for col in num_cols:
    data_sw[col] = data_sw[col].fillna(data_sw[col].median())

# fill NaN for categorical columns using the mode
cat_cols = data_sw.select_dtypes(include=["object"]).columns
for col in cat_cols:
    data_sw[col] = data_sw[col].fillna(data_sw[col].mode()[0])


# %%
#     Create a new column that converts the age ranges to a single number. Drop the age range categorical column
def convert_age(age_range):
    if pd.isnull(age_range):
        return None
    age_mapping = {
        "18-29": 24,
        "30-44": 37,
        "45-60": 53,
        "> 60": 65,
    }
    return age_mapping.get(age_range, None)


data_sw["age_range"] = data_sw["age"].apply(convert_age)
data_sw = data_sw.drop(columns=["age"])


# %%
#     Create a new column that converts the education groupings to a single number. Drop the school categorical column
def convert_education(education):
    if pd.isnull(education):
        return None
    edu_mapping = {
        "Less than high school degree": 1,
        "High school degree": 2,
        "Some college or Associate degree": 3,
        "Bachelor degree": 4,
        "Graduate degree": 5,
    }
    return edu_mapping.get(education, None)


data_sw["education_group"] = data_sw["education"].apply(convert_education)
data_sw = data_sw.drop(columns=["education"])


# %%
#     Create a new column that converts the income ranges to a single number. Drop the income range categorical column
def convert_income(income_range):
    if pd.isnull(income_range):
        return None
    income_mapping = {
        "$0 - $24,999": 125000,
        "$25,000 - $49,999": 375000,
        "$50,000 - $99,999": 75000,
        "$100,000 - $149,999": 125000,
        "$150,000+": 150000,
    }
    return income_mapping.get(income_range, None)


data_sw["income_group"] = data_sw["income"].apply(convert_income)
data_sw = data_sw.drop(columns=["income"])

# %%
#     Create your target (also known as “y” or “label”) column based on the new income range column
data_sw["target_income"] = data_sw["income_group"].apply(
    lambda x: 1 if x > 50000 else 0
)

# %%
#     One-hot encode all remaining categorical columns
data_sw = pd.get_dummies(data_sw, columns=["sw_fan", "gender", "location"])

data_sw.head(10)
# GRAND QUESTION 3
# Validate that the data provided on GitHub lines up with the article by recreating
# 2 of the visuals from the article.

# %%
# Create chart based on the probability that someone has seen a given “Star Wars”
# film, given that they have seen any Star Wars film:

data_sw = data_sw[data_sw["seen_any_sw_film"] == "Yes"]

star_wars_movies_list = [
    "The Phantom Menace",
    "Attack of the Clones",
    "Revenge of the Sith",
    "A New Hope",
    "The Empire Strikes Back",
    "Return of the Jedi",
]

see_movies_list = [
    "seen_ep1",
    "seen_ep2",
    "seen_ep3",
    "seen_ep4",
    "seen_ep5",
    "seen_ep6",
]

seen_film_counts = pd.DataFrame(columns=["Movie", "Percentage"])

percentages = []
for seen in see_movies_list:
    percentage = data_sw[seen].notna().mean() * 100
    percentages.append(percentage)

seen_film_counts = pd.DataFrame(
    {"Movie": star_wars_movies_list, "Percentage": percentages}
)

seen_film_counts["Percentage"] = seen_film_counts["Percentage"].round(0)

# create chart
seen_fig = px.bar(
    seen_film_counts,
    y="Movie",
    x="Percentage",
    title="Which 'Star Wars' Movie Have You Seen?",
    text="Percentage",
    orientation="h",
)

seen_fig.show()

# %%
# create chart do display who shot first? column
# calculate percentage for each category in shot_first column
shot_first_counts = data_sw["shot_first"].value_counts(normalize=True) * 100
shot_first_counts = shot_first_counts.reset_index()
shot_first_counts.columns = ["shot_first", "percentage"]

# round to whole number
shot_first_counts["percentage"] = shot_first_counts["percentage"].round(0)

# create chart
fig = px.bar(
    shot_first_counts,
    y="shot_first",
    x="percentage",
    title="Who Shot First?",
    labels={"shot_first": "Character who shot first", "percentage": "Percentage"},
    text="percentage",
    orientation="h",
)

fig.show()

# %%

# # GRAND QUESTION 4
# Build a machine learning model that predicts whether a person makes more than $50k.
# Describe your model and report the accuracy.

# %%
# split the data in features and target label
X = data_sw.drop(
    columns=[
        "target_income",
        "seen_any_sw_film",
        "seen_ep1",
        "seen_ep2",
        "seen_ep3",
        "seen_ep4",
        "seen_ep5",
        "seen_ep6",
        "fav_han",
        "fav_luke",
        "fav_leia",
        "fav_anakin",
        "fav_obi",
        "fav_emperor",
        "fav_darth",
        "fav_lando",
        "fav_boba",
        "fav_c3po",
        "fav_r2d2",
        "fav_jarjar",
        "fav_padme",
        "fav_yoda",
        "shot_first",
        "expanded_universe_fam",
        "expanded_universe_fan",
        "star_trek_fan",
    ]
)
y = data_sw["target_income"]

# split data in training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# initialize the random forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# show the accuracy
print(f"The accuracy of the Random Forest Classifier is: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
# %%
