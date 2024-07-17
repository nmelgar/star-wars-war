#%%
import pandas as pd

#%%
url = 'https://github.com/fivethirtyeight/data/blob/master/star-wars-survey/StarWars.csv?raw=true'
df = pd.read_csv(url, encoding='latin-1')
print(df.to_string()) 

# %%
df.columns
# %%
