# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:32:32 2020

@author: HariGopal V
"""

Requirements:

python >= 3
pip or conda to install packages
install the necessary packages using req.txt by executing following command:
    -   "pip install -r req.txt"

File-Directory structure:

Heuristic-search-optimization-technique-variants
|_  Benign
    |_  input/images
    |_  output/images
|_  process.py
|_  PSO.py
|_  ReadMe.txt
|_  req.txt

or make sure the files "process.py" and "PSO.py" are under single folder

Executing the project:

1 Modify file "process.py" to meet your requirements and comfort of input and output
2 you can either fix a unique/static input output or give the paths as input everytime or provide them as runtime arguments
3 code for all the 3 options is available in the last portion of the program.
4 Run the file "process.py" using "python>=3" using the command :
    -   "python process.py"
    -   or "python process.py -i <input_path> -o <output_path>" if using command line arguments
5 As the all the thresholding functions doesn't give the best or optimal results the user has to set the threshold for each
  individual image manually when asked by the program graphically in the most simple way.
  The optimal value of the threshold lies where the tumor part is completely seperated and highlighted
  enough to get it segmented.
6 after that the program runs on its own and provide you with outputs which closes when you press enter by selecting them.
7 The results of the GA, AGA & PSO modules is displayed while running and also saved in the output path provided.

Thats it mate. It is done.
Any queries : 9553071182 (mobile/whatsapp)
Thank You

