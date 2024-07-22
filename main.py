# %%
import pandas as pd

# %%
url = "https://github.com/fivethirtyeight/data/blob/master/star-wars-survey/StarWars.csv?raw=true"
data_sw = pd.read_csv(url, encoding="ISO-8859-1")
data_sw = data_sw.drop(index=0)

# %%
data_sw.columns

## GRAND QUESTION 1
# Shorten the column names and clean them up for easier use with pandas. Provide a table
# or list that exemplifies how you fixed the names.
# %%
# %%
# data_sw.columns = [
#     "RespondentID",
#     "Seen_any_SW_film",
#     "SW_fan",
#     "Seen_EP1",
#     "Seen_EP2",
#     "Seen_EP3",
#     "Seen_EP4",
#     "Seen_EP5",
#     "Seen_EP6",
#     "Rank_EP1",
#     "Rank_EP2",
#     "Rank_EP3",
#     "Rank_EP4",
#     "Rank_EP5",
#     "Rank_EP6",
#     "Han_fav",
#     "Luke_fav",
#     "Leia_fav",
#     "Anakin_fav",
#     "Obi_fav",
#     "Emperor_fav",
#     "Darth_fav",
#     "Lando_fav",
#     "Boba_fav",
#     "C3PO_fav",
#     "R2D2_fav",
#     "JarJar_fav",
#     "Padme_fav",
#     "Yoda_fav",
#     "Character_shot_first",
#     "Expanded_universe_familiarized",
#     "Expanded_universe_fan",
#     "Star_Trek_fan",
#     "Gender",
#     "Age",
#     "Income",
#     "Education",
#     "Location",
# ]
data_sw.columns = [
    "respondent_id",
    "seen_any_film",
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

# %%
data_sw.head(15)

## GRAND QUESTION 2
# Clean and format the data so that it can be used in a machine learning model.
# As you format the data, you should complete each item listed below.
# In your final report provide example(s) of the reformatted data with a
# short description of the changes made.

# %%
data_sw["respondent_id"] = range(1, len(data_sw) + 1)

# %%
#     Filter the dataset to respondents that have seen at least one film
data_sw = data_sw[data_sw["seen_any_film"] == "Yes"]

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
data_sw["target_income"] = data_sw["income_target"].apply(
    lambda x: 1 if x > 50000 else 0
)

# %%
#     One-hot encode all remaining categorical columns
data_sw = pd.get_dummies(data_sw, columns=["sw_fan", "gender", "location"])

# GRAND QUESTION 3
# Validate that the data provided on GitHub lines up with the article by recreating
# 2 of the visuals from the article.


# # GRAND QUESTION 4
# Build a machine learning model that predicts whether a person makes more than $50k.
# Describe your model and report the accuracy.

# %%
