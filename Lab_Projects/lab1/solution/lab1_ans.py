# author: ‘Your name'
# student_id: 'Your student ID'
import re


def q1(words):
	print('q1: {:}'.format(words))
	# 1. Print all words beginning with sh
	# YOUR CODE
	w_list = [w for w in words if w.startswith('sh')]
	print(' '.join(w_list))

	# 2. Print all words longer than four characters
	# YOUR CODE
	w_list = [w for w in words if len(w) > 4]
	print(' '.join(w_list))


def q2(file_name):
	print('q2: {:}'.format(file_name))
	# YOUR CODE
	# readfile
	# Your Code
	with open(file_name, 'r') as f:
		lines = f.readlines()

	words = []
	# process text using regex
	for line in lines:
		words += line.strip().split()
		# words += nltk.word_tokenize(line)
		# words += re.split('[^\w]', line)

	print("1. Find words ending in 'ize'")
	# 1. Find words ending in 'ize'
	print(' '.join([w for w in words if w.endswith('ize')]))

	# 2. Find words containing 'z'
	print("2. Find words containing 'z'")
	print(' '.join([w for w in words if 'z' in w]))

	# 3. Containing the sequence of letters "pt"
	print("3. Find words containing 'pt'")
	print(' '.join([w for w in words if 'pt' in w]))

	# 4.
	print("4. Find words that are in titlecase")
	print(' '.join([w for w in words if w.istitle()]))


def q3(line):
	print('q3: {:}'.format(line))
	# YOUR CODE

	# method 1
	# line = re.sub('[^\w]','', line).lower()
	# print(line == line[::-1])

	# method 2
	i,j = 0, -1
	line = line.lower()
	for i in range(len(line)//2):
		wi = line[i]
		if not wi.isalpha():
			i += 1
			continue
		while not line[j].isalpha():
			j -= 1
		wj = line[j]
		if wi != wj:
			print(False)
			return
		j -= 1
	print(True)


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