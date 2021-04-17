import pickle

#with open("iwslt14.tokenized.de-en/tmp/train.tags.de-en.tok.en", "r") as file:
with open("../data/iwslt14.tokenized.de-en/vocab.en.pkl", "rb") as file, open("../data/iwslt14.tokenized.de-en/dict.en.txt", "w") as dictfile:
	data = pickle.load(file)
	for key in data:
		dictfile.write(f"{key} {data[key]}\n")
		#print(f"{key} {data[key]}")

#print(str(data))
#print(data)

with open("../data/iwslt14.tokenized.de-en/vocab.de.pkl", "rb") as file, open("../data/iwslt14.tokenized.de-en/dict.de.txt", "w") as dictfile:
	data = pickle.load(file)
	for key in data:
		dictfile.write(f"{key} {data[key]}\n")
		#print(f"{key} {data[key]}")

#print(str(data))
#print(data)