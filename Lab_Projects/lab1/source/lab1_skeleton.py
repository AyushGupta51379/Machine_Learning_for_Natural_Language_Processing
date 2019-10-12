# author: ‘Your name'
# student_id: 'Your student ID'

def q1(words):
    print('q1: {:}'.format(words))
    # 1. Print all words beginning with sh
    # YOUR CODE
    
    # 2. Print all words longer than four characters
    # List Comprehensions
    # YOUR CODE

def q2(file_name):
    print('q2: {:}'.format(file_name))
    # YOUR CODE
    # readfile
    # Your Code

    words = []
    # process text using regex or nltk.word_tokenize()
    # Your Code
    
    # 1. Find words ending in 'ize'
    print("1. Find words ending in 'ize'")
    # Your Code

    # 2. Find words containing 'z'
    print("2. Find words containing 'z'")
    # Your Code

    # 3. Containing the sequence of letters "pt"
    print("3. Find words containing 'pt'")
    # Your Code

    # 4. Find words that are in titlecase
    print("4. Find words that are in titlecase")


def q3(line):
    print('q3: {:}'.format(line))
    # YOUR CODE


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
