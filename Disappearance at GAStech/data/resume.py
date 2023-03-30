import os
import nltk
from contextlib import redirect_stdout

from rake_nltk import Rake
rake_nltk_var = Rake()

in_dir_path = r'D:/roy/selected/TU/books/visual_analytics/Disappearance at GAStech/data/resumes/txt versions'
out_dir_path = r'D:/roy/selected/TU/books/visual_analytics/Disappearance at GAStech/data/output_resume'

keywords = [" Armed forces of Kronos", "Protectors of Kronos", "Kronos Armed"]

for file_name in os.listdir(in_dir_path):
    if file_name.endswith('.txt'):
        input_path = os.path.join(in_dir_path, file_name)
        output_path = os.path.join(out_dir_path, file_name)
        with open(input_path, 'r') as infile:
            with open(output_path, 'w') as outfile:
                text = infile.read()
                print(type(text))
                for k in keywords: 
                    if k.lower() in text.lower():
                        outfile.write("%s\n"%text)
                        print("File moved succesfully")
                    else:
                        print("Unsuccesfull")