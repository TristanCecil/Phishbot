import re
import json
#import frequency_Counter
import utils2

def buildVocab():
    global tokens
    tokens = [1,2,3]
    file = open("tokenizer.tok-vocab.json", "r", encoding="utf8", errors='namereplace')
    #examples = open("htmlCodePerLine.txt", "r", encoding="utf8", errors="namereplace")
    output = open("vocab.txt", "wb")
    #frequencyOutput = open("frequencies.txt", "wb")
    if file == -1:
        print("File IO Error")

    print("Starting token scan")
   # regex = '".*:"'

    line_string = file.read()
    data = json.loads(line_string)


    #matches = []
    keys = data.keys()
    tokens = keys
    temp = "\n"
    for k in keys:
        utils2.vocab[k] = 0
        output.write(k.encode())
        output.write(temp.encode())
    """
    while (line_string):
        matches = re.findall(regex, line_string)
        for m in matches:
            output.write(m.encode())
            output.write(temp)
            
    """

    output.close()
    file.close()
    """
    htmlCode = examples.readline()
    while htmlCode:
        fc.tokenfreqs(htmlCode)
        htmlCode = examples.readline()
    """
    print("Please check vocab.txt to see that all tokens have been processed correctly")




