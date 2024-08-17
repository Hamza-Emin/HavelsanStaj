import pandas
import string
import re

# Load CSV file
dframe = pandas.read_csv("vodafone_data.csv", encoding='utf-8', on_bad_lines='skip', sep=';')
print(dframe)

# Clean text function
def clean_text(text):
    text = " ".join(str(text).split())
    text = text.lower()
    text = text.replace("\\n", " ")
    text = re.sub("[0-9]+", "", text)
    text = re.sub("%|(|)|-", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return "".join(text)

# Apply cleaning to the Explanation column
dframe["Clean"] = dframe.apply(lambda row: clean_text(row["Explanation"]), axis=1)

# Function to label aspects
def label_aspects(text):
    aspects = []
    
    if "vodafone" in text:
        aspects.append("vodafone")

    if "turkcell" in text:
        aspects.append("turkcell")

    if "turk telekom" in text or "telekom" in text:
        aspects.append("telekom")

    list_of_pricing = ['tl', 'pahalı', 'yüksek', 'az', 'ucuz', 'indirim']
    for word in list_of_pricing:
        if word in text:
            aspects.append('Price')
            break

    signal_problems = ['çekiyordu', 'çekmiyor', 'sinyal', 'duyulmuyor', 'çek']
    for word in signal_problems:
        if word in text:
            aspects.append('Sinyal')
            break

    comparison = ['daha iyi', 'daha fazla', 'daha ucuz', 'daha kaliteli']
    for word in comparison:
        if word in text:
            aspects.append('Comparison')
            break

    # If no aspects were found, return ['None']
    if not aspects:
        aspects.append('None')
        
    return aspects

# Apply aspect labeling to the Clean column
dframe["Aspects"] = dframe["Clean"].apply(label_aspects)

# Print the resulting DataFrame
print(dframe)

# Save to a new CSV file
dframe.to_csv("output.csv", sep=';', encoding='utf-8')
