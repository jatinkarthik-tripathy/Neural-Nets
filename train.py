from colorama import Fore, Style, init
init()

if __name__ == '__main__':
	try:
		with open('data/vocab', 'r', encoding='utf8') as f:
			vocab = [word.strip('\n') for word in f]
		with open('data/train.from', 'r', encoding='utf8') as f:
			ques = [word.strip('\n') for word in f]
		with open('data/train.to', 'r', encoding='utf8') as f:
			ans = [word.strip('\n') for word in f]
	except FileNotFoundError:
		print(f'{Fore.RED}Please format data before training{Style.RESET_ALL}')

	inp_len = len(ques)
	out_len = len(ans)