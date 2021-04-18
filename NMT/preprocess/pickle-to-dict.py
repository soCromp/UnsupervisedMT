import pickle
import os
import sys

#with open("iwslt14.tokenized.de-en/tmp/train.tags.de-en.tok.en", "r") as file:
with open(os.path.join(sys.argv[1], "vocab.en.pkl"), "rb") as file, open(os.path.join(sys.argv[1], "dict.en.txt"), "w") as dictfile:
	data = pickle.load(file)
	for key in data:
		dictfile.write(f"{key} {data[key]}\n")
		#print(f"{key} {data[key]}")

#print(str(data))
#print(data)

with open(os.path.join(sys.argv[1], "vocab.de.pkl"), "rb") as file, open(os.path.join(sys.argv[1], "dict.de.txt"), "w") as dictfile:
	data = pickle.load(file)
	for key in data:
		dictfile.write(f"{key} {data[key]}\n")
		#print(f"{key} {data[key]}")

#print(str(data))
#print(data)