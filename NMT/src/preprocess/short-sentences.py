#!/usr/bin/env python
import sys
import os

def shrink(data_path, lang1, lang2, sent_maxlen=30):
	f1_lines = []
	f2_lines = []
	data_files = ["train", "valid", "test"]

	for file in data_files:
		lang1_file_name = file + "." + lang1
		lang2_file_name = file + "." + lang2
		print(f"File names: {lang1_file_name}, {lang2_file_name}")

		with open(os.path.join(data_path, lang1_file_name), "r") as file1, open(os.path.join(data_path, lang2_file_name), "r") as file2, \
		     open(os.path.join(data_path, ("old_"+lang1_file_name)), "w") as f1_write, open(os.path.join(data_path, ("old_"+lang2_file_name)), "w") as f2_write:

			f1_lines = file1.readlines()
			f2_lines = file2.readlines()
			all_f1 = file1.read()
			all_f2 = file2.read()

			f1_write.write(all_f1)
			f2_write.write(all_f2)

		if len(f1_lines) != len(f2_lines):
			print("~Internally screaming~")

		print(f"Eliminating lines over {sent_maxlen} words in {lang1_file_name}, {lang2_file_name}")
		with open(os.path.join(data_path, lang1_file_name), "w") as file1, open(os.path.join(data_path, lang2_file_name), "w") as file2:
			for x in range(len(f1_lines)):
				#print("Help: ", type(len(f1_lines[x].split(' '))), type(sent_maxlen), type(len(f2_lines[x].split(' '))), type(sent_maxlen))
				if len(f1_lines[x].split(' ')) <= sent_maxlen and len(f2_lines[x].split(' ')) <= sent_maxlen:
					file1.write(f1_lines[x])
					file2.write(f2_lines[x])

if __name__ == "__main__":
	''' 
	args are as follows: 
	
	first arg = data path (directory within the data directory to get the train, valid, test files)
	second arg = langs
	third arg = max sentence length
	
	'''
	args = sys.argv[1:]

	#data_folder = os.path.join(os.getcwd(), "../data")
	#data_path = os.path.join(data_folder, args[0])
	data_path = os.path.join("../data", args[0])

	langs = args[1]
	lang1, lang2 = langs.split("-")

	sent_maxlen = int(args[2])

	print(f"args: {data_path} {lang1} {lang2} {sent_maxlen}")

	shrink(data_path, lang1, lang2, sent_maxlen)