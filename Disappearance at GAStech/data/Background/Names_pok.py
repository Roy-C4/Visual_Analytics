import re
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Regular expression pattern for names
name_pattern = r"[A-Z][a-z]+(?: [A-Z][a-z]+)*"

in_dir_path = r'D:\roy\selected\TU\books\visual_analytics\New folder (2)\Disappearance at GAStech\data\HistoricalDocuments\txt versions'
out_dir_path = r'D:\roy\selected\TU\books\visual_analytics\New folder (2)\Disappearance at GAStech\data\Background\names_pok'

def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())                                        # Tokenize the text into words and convert to lowercase
    stop_words = set(stopwords.words('english'))                                # Get a set of stopwords for English language
    words = [word for word in tokens if word.isalpha()                          # Remove non-alphabetic words and stopwords from the tokenized words
             and word not in stop_words]
    cleaned_text = ' '.join(words)                                               # Join the remaining words with a single space in between
    cleaned_text = ' '.join(cleaned_text.split())                                # Replace any whitespace characters with a single space
    return cleaned_text


# Open input and output files
for file_name in os.listdir(in_dir_path):
    if file_name.endswith('.txt'):
        input_path = os.path.join(in_dir_path, file_name)
        output_path = os.path.join(out_dir_path, file_name)
        with open(input_path, 'r', encoding='utf-8') as infile:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                # Loop over lines in input file
                for line in infile:
                    # Use regular expression to find names
                    names = re.findall(name_pattern, line)
                    # Write names to output file
                    for name in names:
                        outfile.write(name + "\n")

