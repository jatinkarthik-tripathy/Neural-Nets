import numpy as np
import json
from colorama import Fore, Style, init
init()

if __name__ == '__main__':
	# reading the data
	try:
		with open('data/train.from', 'r', encoding='utf8') as f:
			ques = [word.strip('\n') for word in f]
		with open('data/train.to', 'r', encoding='utf8') as f:
			ans = [word.strip('\n') for word in f]
		with open('data/input_vocab.json', 'r', encoding='utf8') as f:
			input_vocab = json.load(f)
		with open('data/output_vocab.json', 'r', encoding='utf8') as f:
			output_vocab = json.load(f)
	except FileNotFoundError:
		print(f'{Fore.RED}Please format data before training{Style.RESET_ALL}')

	inp_len = len(input_vocab)
	out_len = len(output_vocab)

	max_inp_len = max([len(word) for word in ques])
	max_out_len = max([len(word) for word in ans])
	print(f'num of elements for encoder input: {len(ques)*max_inp_len*inp_len}')

	encoder_inp = np.zeros((len(ques), max_inp_len, inp_len), dtype='bool')
	decoder_inp = np.zeros((len(ans), max_out_len, out_len), dtype='float32')
	decoder_out = np.zeros((len(ans), max_out_len, out_len), dtype='float32')

	for i, (q, a) in enumerate(zip(ques, ans)):
		for t, word in enumerate(q):
			print(t, word)

