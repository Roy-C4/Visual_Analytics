import os
import nltk
from contextlib import redirect_stdout

from rake_nltk import Rake
rake_nltk_var = Rake()

in_dir_path = r'D:/roy/selected/TU/books/visual_analytics/Disappearance at GAStech/data/articles'
out_dir_path = r'D:/roy/selected/TU/books/visual_analytics/Disappearance at GAStech/data/output_article'

for file_name in os.listdir(in_dir_path):
    if file_name.endswith('.txt'):
        input_path = os.path.join(in_dir_path, file_name)
        output_path = os.path.join(out_dir_path, file_name)
        with open(input_path, 'r') as infile:
            with open(output_path, 'w') as outfile:
                text = infile.read()
                rake_nltk_var.extract_keywords_from_text(text)
                keyword_extracted = rake_nltk_var.get_ranked_phrases()
                for item in keyword_extracted:
                    outfile.write("%s\n"%item)
            