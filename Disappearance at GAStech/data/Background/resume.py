import os
import nltk
from contextlib import redirect_stdout

from rake_nltk import Rake
rake_nltk_var = Rake()

in_dir_path = r'D:\roy\selected\TU\books\visual_analytics\Disappearance at GAStech\data\Background\resumes\txt_versions'
out_dir_path = r'D:\roy\selected\TU\books\visual_analytics\Disappearance at GAStech\data\Background\output_resume'

keywords = [" Armed forces of Kronos", "Protectors of Kronos", "Kronos Armed", "Armed", "Forces", "Protectors", "PoK"]

for file_name in os.listdir(in_dir_path):
    if file_name.endswith('.txt'):
        input_path = os.path.join(in_dir_path, file_name)
        output_path = os.path.join(out_dir_path, file_name)
        with open(input_path, 'r') as infile:
            with open(output_path, 'w') as outfile:
                text = infile.read()
                for j in text.split():
                    if j in keywords:
                        print("keyword present", j)
                        outfile.write("%s\n"%text)
                        print("File moved succesfully")
                    else:
                        print("Unsuccesfull")

for filename in os.listdir(out_dir_path):
    file_path = os.path.join(out_dir_path, filename)
    
    # check if the file is a text file and is  not a directory
    if filename.endswith('.txt') and os.path.isfile(file_path):
        
        # check if the file is empty
        if os.stat(file_path).st_size == 0:
            os.remove(file_path) # delete the file
            print(f"{filename} has been deleted")