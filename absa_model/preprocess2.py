import re
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import pandas as pd


df = pd.read_csv('output.csv', on_bad_lines='skip', sep=';')
df = df.drop(columns='Unnamed: 0', axis=1)


def clean_aspects(aspects_str):
   
    aspects_str = re.sub(r"[\[\]'\"\s]", "", aspects_str)
    
    aspects_list = aspects_str.split(',')
    return aspects_list


df['Aspects'] = df['Aspects'].apply(clean_aspects)


mlb = MultiLabelBinarizer()
aspect_encoded = mlb.fit_transform(df['Aspects'])

aspect_encoded_df = pd.DataFrame(aspect_encoded, columns=mlb.classes_)


df = pd.concat([df, aspect_encoded_df], axis=1)


joblib.dump(mlb, 'mlb.pkl')


print("Corrected MLB Classes:", mlb.classes_)


df = df.to_csv('output_encoded.csv', sep=';', index=False)


#print(df['Aspects'].tolist())
