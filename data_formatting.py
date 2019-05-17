import json
from colorama import Fore, Style, init
init()


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def vocab_build():
	#read json file
	with open('data/clean/conversations.json', 'r') as f:
	    convs = json.load(f)

	ques = []
	ans = []
	vocab = []
	counter = 0
	tot = len(convs)
	print(f'Total conversations: {tot}')
	printProgressBar(0, tot, prefix = 'Progress:', suffix = 'Complete', length = 50)
	for q, a in convs.items():
		q = q.strip().replace('.', '').replace('\t', '')
		a = a.strip().replace('.', '').replace('\t', '')

		ques.append(q)
		ans.append(a)

		for word in q.split(' '):
			if word not in vocab:
				vocab.append(word)
		for word in a.split(' '):
			if word not in vocab:
				vocab.append(word)
		counter += 1
		printProgressBar(counter, tot, prefix = 'Progress:', suffix = 'Complete', length = 50)

	vocab = sorted(vocab)
	with open('data/vocab', 'w', encoding='utf8') as f:
		for word in vocab:
			f.write(word+'\n')
	with open('data/train.from', 'w', encoding='utf8') as f:
		for q in ques:
			f.write(q+'\n')
	with open('data/train.to', 'w', encoding='utf8') as f:
		for a in ans:
			f.write(a+'\n')
	return vocab, ques, ans

if __name__ == '__main__':
	try:
		with open('data/vocab', 'r', encoding='utf8') as f:
			vocab = [word.strip('\n') for word in f]
		with open('data/train.from', 'r', encoding='utf8') as f:
			ques = [word.strip('\n') for word in f]
		with open('data/train.to', 'r', encoding='utf8') as f:
			ans = [word.strip('\n') for word in f]
	except FileNotFoundError:
		vocab, ques, ans = vocab_build()
	print(f'{Fore.GREEN}Data Formatting Complete{Style.RESET_ALL}')
