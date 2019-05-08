import json

def create_dict():
	id_dict = {}
	with open('data/raw/movie_lines.txt', 'r') as f:
		lines = f.read().split('\n')
		counter = 0
		for line in lines:
			line_part = line.split(' +++$+++ ')
			if len(line_part) == 5:
				id_dict[line_part[0]] = line_part[4]
				if counter%10000 == 0:
					print (line_part[0], id_dict[line_part[0]])
				counter += 1
	with open('data/clean/id_dict.json', 'w') as f:
		json.dump(id_dict, f, sort_keys=True, indent=4)

create_dict()