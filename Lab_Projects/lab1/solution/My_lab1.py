# author: ‘GUPTA, Ayush' | 'Ayush GUPTA'
import re

def q1(words):
    print('q1: {:}'.format(words))
    # 1. Print all words beginning with sh
    # YOUR CODE
    for w in words:
        if w.startswith("sh"):
            print (w, end= " ")
    print ()

    # Can use other ways than what is taught in labs, this is the quick way but we can definitely try other methods

    
    # 2. Print all words longer than four characters
    # List Comprehensions
    # YOUR CODE

    word_list = [w for w in words if len(w)>4]
    print (' '.join(word_list))
    

def q2(file_name):
    print('q2: {:}'.format(file_name))
    # YOUR CODE
    # readfile
    # Your Code

    with open(file_name, 'r') as f:
        lines = f.readlines()  

    words = []
    # process text using regex or nltk.word_tokenize()
    # Your Code
    words = [w for line in lines for w in line.split()]
    
    # 1. Find words ending in 'ize'
    print("1. Find words ending in 'ize'")
    # Your Code
    print(" ".join( w for w in words if w.endswith('ize')))
    
    # 2. Find words containing 'z'
    print("2. Find words containing 'z'")
    # Your Code
    print(" ".join( w for w in words if 'z' in w))

    # 3. Containing the sequence of letters "pt"
    print("3. Find words containing 'pt'")
    # Your Code
    print(" ".join( w for w in words if 'pt' in w))

    # 4. Find words that are in titlecase
    print("4. Find words that are in titlecase")
    print(" ".join( w for w in words if w.istitle()))

def q3(line):
    print('q3: {:}'.format(line))
    # YOUR CODE
    # method 1
    """    line = line.lower()
    i,j = 0, len(line)-1
    for i in range(len(line)//2):
        ai = line[i]
        if not ai.isalpha():
            continue
        while not line[j].isalpha():
            j-=1
            continue
        aj = line[j]
        if ai != aj:
            print(False)
            return
        j-= 1

    print (True)
    """
    #method 2
    line = line.lower()
    line = re.sub('[^\w]', '', line)
    print(line == line[::-1])
    
        
    


if __name__ == '__main__':
    q1_input = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
    q1(q1_input)

    print()
    q2_input = 'lab1_text.txt'
    q2(q2_input)

    print()
    q3_input = ['A man, a plan, a canal: Panama', 'race a car', 'raca a                              car']
    for q3_in in q3_input:
        q3(q3_in)
