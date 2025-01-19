# precompute_sparse_vectors.py
# Given the output data from preprocess.py in doc_to_words and 
# word_to_docs, precompute the sparse vector representations of TF-IDF
# and BM25.
# Python 3.11
# Windows/MacOS/Linux


import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import multiprocessing as mp
import os
import shutil
import string
from typing import Dict, List, Set

from bs4 import BeautifulSoup
import msgpack
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_data_from_msgpack(path: str) -> Dict:
	'''
	Load a data file (to dictionary) from msgpack file given the path.
	@param: path (str), the path of the data file that is to be loaded.
	@return: Returns a python dictionary containing the structured data
		from the loaded data file.
	'''
	with open(path, 'rb') as f:
		byte_data = f.read()

	return msgpack.unpackb(byte_data)


def load_data_from_json(path: str) -> Dict:
	'''
	Load a data file (to dictionary) from either a file given the path.
	@param: path (str), the path of the data file that is to be loaded.
	@return: Returns a python dictionary containing the structured data
		from the loaded data file.
	'''
	with open(path, "r") as f:
		return json.load(f)
	

def load_data_file(path: str, use_json: bool = False) -> Dict:
	'''
	Load a data file (to dictionary) from either a JSON or msgpack file
		given the path.
	@param: path (str), the path of the data file that is to be loaded.
	@param: use_json (bool), whether to load the data file using JSON 
		msgpack (default is False).
	@return: Returns a python dictionary containing the structured data
		from the loaded data file.
	'''
	if use_json:
		return load_data_from_json(path)
	return load_data_from_msgpack(path)


def write_data_to_msgpack(path: str, data: Dict) -> None:
	'''
	Write data (dictionary) to a msgpack file given the path.
	@param: path (str), the path of the data file that is to be 
		written/created.
	@param: data (Dict), the structured data (dictionary) that is to be
		written to the file.
	@return: returns nothing.
	'''
	with open(path, 'wb+') as f:
		packed = msgpack.packb(data)
		f.write(packed)


def write_data_to_json(path: str, data: Dict) -> None:
	'''
	Write data (dictionary) to a json file given the path.
	@param: path (str), the path of the data file that is to be 
		written/created.
	@param: data (Dict), the structured data (dictionary) that is to be
		written to the file.
	@return: returns nothing.
	'''
	with open(path, "w+") as f:
		json.dump(data, f, indent=4)


def write_data_file(path: str, data: Dict, use_json: bool = False) -> None:
	'''
	Write data (dictionary) to either a JSON or msgpack file given the
		path.
	@param: path (str), the path of the data file that is to be loaded.
	@param: data (Dict), the structured data (dictionary) that is to be
		written to the file.
	@param: use_json (bool), whether to write the data file to a JSON 
		or msgpack (default is False).
	@return: returns nothing.
	'''
	if use_json:
		write_data_to_json(path, data)
	else:
		write_data_to_msgpack(path, data)


def clear_folder(folder_path: str) -> None:
	'''
	Clear the contents of the given folder.
	@param: folder_path (str), the (valid) folder that should be 
		emptied of its contents.
	@return: returns nothing.
	'''
	# Iterate through each item found in the folder path.
	for item in os.listdir(folder_path):
		# Get the full path of the item.
		item_path = os.path.join(folder_path, item)

		try:
			# Try and delete the item (folder or file).
			if os.path.isfile(item_path) or os.path.islink(item_path):
				os.unlink(item_path)  # Remove files or symlinks
				print(f"Deleted file: {item_path}")
			elif os.path.isdir(item_path):
				shutil.rmtree(item_path)  # Remove directories and their contents
				print(f"Deleted folder: {item_path}")
		except Exception as e:
			print(f"Failed to delete {item_path}. Reason: {e}")


def clear_staging_folders(staging_folders: List[str]) -> None:
	'''
	Clear the contents of all staging folders
	@param: staging_folders (List[str]), the list of all (valid) 
		staging folders that should be emptied of their contents.
	@return: returns nothing.
	'''
	for folder in staging_folders:
		assert os.path.exists(folder)
		clear_folder(folder)


def isolate_invalid_articles(pages: List[str]) -> List[str]:
	'''
	Isolate the invalid articles away from the valid ones.
	@param: pages (List[str]), the list of articles parsed by 
		beautifulsoup that need to be analyzed.
	@return, returns a list containing the SHA1 strings of all invalid
		articles from the input list.
	'''
	# Initialize a list to store the article SHA1's for all invalid
	# articles.
	redirect_shas = list()

	# Iterate through each article.
	for page_str in tqdm(pages):
		# Parse page with beautifulsoup.
		page = BeautifulSoup(page_str, "lxml")

		# Isolate the article/page's SHA1.
		sha1_tag = page.find("sha1")

		# Skip articles that don't have a SHA1 (should not be 
		# possible but you never know).
		if sha1_tag is None:
			continue

		# Clean article SHA1 text.
		article_sha1 = sha1_tag.get_text()
		article_sha1 = article_sha1.replace(" ", "").replace("\n", "")

		# Isolate the article/page's redirect tag.
		redirect_tag = page.find("redirect")

		# Skip articles that have a redirect tag (they have no 
		# useful information in them).
		if redirect_tag is not None:
			redirect_shas.append(article_sha1)

	# Return the list of invalid article SHAs.
	return redirect_shas


def get_document_lengths(doc_to_words: Dict[str, Dict[str, int]]) -> List[int]:
	'''
	Given the list of documents, retrieve the lengths of each one.
	@param: doc_to_word (Dict[str, Dict[str, int]]), the map of all
		documents in a file and their respective word frequency 
		mappings as well.
	@return: returns a list containing the document lengths (List[int]).
	'''
	# Initialize a list containing the document lengths.
	document_lengths = list()

	# Iterate through the documents.
	for article in tqdm(doc_to_words.keys()):
		# Compute the document length by taking the sum of all word
		# frequencies in the document.
		doc_length = sum(
			[value for value in doc_to_words[article].values()]
		)

		# Addthe document length to the return list.
		document_lengths.append(doc_length)

	# Return the list of document lengths.
	return document_lengths


def compute_idf(
	word_to_doc_files: List[str], corpus_size: int, use_json: bool = False
) -> Dict[str, float]:
	'''
	Compute the inverse document frequency for every word in the 
		corpus.
	@param: word_to_doc_files (List[str]), the list of paths for
		all the word to document mapping files for each text file.
	@param: corpus_size (int), the number of documents that exist 
		in the corpus.
	@param: use_json (bool), whether to read the data file from a 
		JSON or msgpack (default is False).
	@return, returns a dictionary mapping every word to its respective
		inverse document frequency.
	'''
	# Aggregate the word count across all documents.
	word_count = dict()
	for word_to_doc_file in tqdm(word_to_doc_files, "Aggregating word counts"):
		word_to_docs = load_data_file(word_to_doc_file, use_json=use_json)

		# Perform set operations to identify words not yet found. 
		# Update the word count dictionary to contain a value of 0 for
		# all those "new" unseen words. This will mean that every word
		# is already initialize when it comes time to iterate through 
		# the word to doc dictionary.
		missing_words = set(word_to_docs.keys())\
			.difference(set(word_count.keys()))
		word_count.update({word: 0 for word in missing_words})

		for word in list(word_to_docs.keys()):
			word_count[word] += word_to_docs[word]
			# if word not in list(word_count.keys()):
			# 	word_count[word] = word_to_docs[word]
			# else:
			# 	word_count[word] += word_to_docs[word]

	# Compute the inverse documemnt frequency for all words.
	word_idf = dict()
	for word in tqdm(list(word_count.keys()), f"Computing IDF"):
		word_idf[word] = math.log(corpus_size / word_count[word])

	# Return the IDF dictionary.
	return word_idf


def compute_sparse_vectors(
	doc_to_words: Dict[str, Dict[str, int]], 
	idf_df: pd.DataFrame, 
	k1: float,
	b: float,
	avg_doc_len: float,
) -> pd.DataFrame:
	'''
	Compute the sparse vector values TF-IDF and BM25 for each document,
		word pair.
	@param: doc_to_words (Dict[str, Dict[str, int]]), the word 
		frequencies for each document in a file.
	@param: idf_df (pd.DataFrame), dataframe containing the word to 
		inverse document frequency mapping for all words in the corpus.
	@param: k1 (float), a hyperparamter necessary for computing BM25.
	@param: b (float), a hyperparamter necessary for computing BM25.
	@param: avg_doc_len (float), the average length of all documents in
		the corpus, needed for computing BM25.
	@return: Returns a pandas DataFrame containing the document (path), 
		word, document level word frequency, document length, word 
		inverse document frequency, document level TF-IDF, and the 
		document level word BM25 score.
	'''
	# Process is vectorized with pandas. Flatten doc to words.
	data = [
		{"doc": doc, "word": word, "tf": tf}
		for doc, word_freq in doc_to_words.items()
		for word, tf in word_freq.items()
	]
	doc_word_df = pd.DataFrame(data)

	# Compute document lengths and merge.
	doc_lengths = (
		doc_word_df.groupby("doc")["tf"]
		.sum()
		.rename("doc_len")
		.reset_index()
	)
	doc_word_df = doc_word_df.merge(
		doc_lengths, on="doc", how="left"
	)

	# Merge with `idf_df` to include the `idf` values
	doc_word_df = doc_word_df.merge(
		idf_df, on="word", how="left"
	)

	# Compute `TF-IDF` and `BM25` scores
	doc_word_df["tf_idf"] = doc_word_df["tf"] * doc_word_df["idf"]
	doc_word_df["bm25"] = (
		doc_word_df["idf"]
		* doc_word_df["tf"]
		* (k1 + 1)
		/ (
			doc_word_df["tf"]
			+ k1 * (
				1 - b + b * (doc_word_df["doc_len"] / avg_doc_len)
			)
		)
	)

	# Return dataframe.
	return doc_word_df


def build_trie(doc_to_word: Dict[str, List[int]]) -> Dict:
	trie = {}
	for word, numbers in doc_to_word.items():
		current = trie
		for char in word:
			current = current.setdefault(char, dict())

		# Add the number to the set at the end of the word
		current.setdefault("doc_id", list()).extend(numbers)
	
	return trie


def search_trie(trie: Dict, word: str) -> Set | None:
	current = trie
	for char in word:
		if char not in current:
			return None
		
		current = current[char]

	return current.get("doc_id")  # Return the set of numbers if the word ends here


def main():
	# Initialize argument parser.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--use_json",
		action="store_true",
		help="Whether to read from JSON or msgpack files. Default is false/not specified."
	)
	parser.add_argument(
		"--num_proc",
		type=int,
		default=1,
		help="How many processors to use. Default is 1."
	)
	parser.add_argument(
		"--num_thread",
		type=int,
		default=1,
		help="How many threads to use. Default is 1."
	)
	parser.add_argument(
		"--max_depth",
		type=int,
		default=1,
		help="How deep should the graph traversal go across links. Default is 1/not specified."
	)
	parser.add_argument(
		"--clear-staging",
		choices=["before", "after", "both", "neither"],
        default="neither",
        help="Specify when to clean up staging: 'before' processing, 'after' processing, 'both' (before and after processing), or 'neither' (default)."
	) # before, after, both, neither

	# Parse arguments.
	args = parser.parse_args()
	use_json = args.use_json
	extension = ".json" if use_json else ".msgpack"
	num_proc = args.num_proc
	num_thread = args.num_thread
	clear_staging = args.clear_staging
	max_depth = args.max_depth

	assert max_depth > 0, \
		f"Invalid --max_depth value was passed in (must be > 0, recieved {max_depth})"

	num_cpus = min(mp.cpu_count(), num_proc)
	max_workers = num_cpus if num_proc > 1 else num_thread

	# Load config file and isolate key variables.
	if not os.path.exists("config.json"):
		print("Could not detect required file config.json in current path.")
		print("Exiting program.")
		exit(1)

	with open("config.json", "r") as f:
		config = json.load(f)

	# Isolate paths.
	preprocessing_paths = config["preprocessing"]

	# Output and staging folder paths.
	idf_staging = os.path.join(
		preprocessing_paths["staging_idf_path"], 
		f"depth_{max_depth}"
	)
	corpus_staging = os.path.join(
		preprocessing_paths["staging_corpus_path"], 
		f"depth_{max_depth}"
	)
	output_folder = os.path.join(
		preprocessing_paths["sparse_vector_path"],
		f"depth_{max_depth}"
	)
	trie_folder = os.path.join(
		preprocessing_paths["trie_path"],
		f"depth_{max_depth}"
	)

	# Doc to word and word to doc folder paths.
	doc_to_words_folder = os.path.join(
		preprocessing_paths["doc_to_words_path"], 
		f"depth_{max_depth}"
	)
	word_to_docs_folder = os.path.join(
		preprocessing_paths["word_to_docs_path"],
		f"depth_{max_depth}"
	)

	# Validate word to doc and doc to word folder paths exist.
	d2w_folder_exists = os.path.exists(doc_to_words_folder)
	w2d_folder_exists = os.path.exists(word_to_docs_folder)
	if not d2w_folder_exists or not w2d_folder_exists:
		print(f"Error: Could not find path {doc_to_words_folder} or {word_to_docs_folder}")
		print("Make sure to run preprocess.py before this script.")
		exit(1)

	# Validate word to doc and doc to word folder paths are populated.
	doc_to_words_files = [
		os.path.join(doc_to_words_folder, file)
		for file in os.listdir(doc_to_words_folder)
		if file.endswith(extension)
	]
	word_to_docs_files = [
		os.path.join(word_to_docs_folder, file)
		for file in os.listdir(word_to_docs_folder)
		if file.endswith(extension)
	]
	num_w2d_items = len(doc_to_words_files)
	num_d2w_items = len(word_to_docs_files)
	if num_w2d_items == 0 or num_d2w_items == 0:
		print(f"Error: Path {doc_to_words_folder} or {word_to_docs_folder} were found to be empty")
		print("Make sure to run preprocess.py before this script.")
		exit(1)

	# Create the output folders if necessary.
	if not os.path.exists(idf_staging):
		os.makedirs(idf_staging, exist_ok=True)

	if not os.path.exists(corpus_staging):
		os.makedirs(corpus_staging, exist_ok=True)
		
	if not os.path.exists(output_folder):
		os.makedirs(output_folder, exist_ok=True)

	if not os.path.exists(trie_folder):
		os.makedirs(trie_folder, exist_ok=True)

	# Clear staging (if applicable).
	if clear_staging in ["before", "both"]:
		clear_staging_folders(
			[idf_staging, corpus_staging, corpus_staging]
		)

	###################################################################
	# Stage 1: Corpus Statistics mpute Average Document Length
	###################################################################
	print("Computing corpus level statistics...")

	# Target corpus statistics:
	# corpus size (number of documents)
	# average document length (for BM25)

	# Path to output JSON.
	corpus_path = os.path.join(corpus_staging, "corpus_stats.json")

	# Perform processing if the end json file is not available.
	if not os.path.exists(corpus_path):
		# Initialize a list to contain all document lengths.
		document_lengths = list()

		# Iterate through each file.
		for idx, file in enumerate(doc_to_words_files):
			# Isolate the basename and print out the current file and
			# its position.
			basename = os.path.basename(file)
			print(f"Processing {basename} ({idx + 1}/{len(doc_to_words_files)})")

			# Load the doc to words map.
			doc_to_words = load_data_file(file, use_json)

			# Remove the invalid articles from the keys (effectively
			# ignore invalid articles).
			valid_articles = list(doc_to_words.keys())

			# Chunk the data to enable concurrency/parallelism.
			chunk_size = math.ceil(len(valid_articles) / max_workers)
			articles_list = [
				valid_articles[i:i + chunk_size] 
				for i in range(0, len(valid_articles), chunk_size)
			]
			doc_to_words_list = [
				{
					article: doc_to_words[article] 
					for article in article_sublist
				}
				for article_sublist in articles_list
			]
			args_list = [
				# (articles_sublist) 
				# for articles_sublist in articles_list
				(doc_to_words_subdict,)
				for doc_to_words_subdict in doc_to_words_list
			]
			
			# Get every (valid) document's length.
			if num_proc > 1:
				with mp.Pool(max_workers) as pool:
					results = pool.starmap(
						get_document_lengths, args_list
					)
					for result in results:
						document_lengths += result
			else:
				with ThreadPoolExecutor(max_workers) as executor:
					results = executor.map(
						lambda args: get_document_lengths(*args),
						args_list
					)
					for result in results:
						document_lengths += result
		
		# Compute the average document length and the corpus size.
		corpus_size = len(document_lengths)
		avg_doc_len = sum(document_lengths) / corpus_size

		# Store in staging.
		with open(corpus_path, "w+") as f:
			json.dump(
				{
					"corpus_size": corpus_size, 
					"avg_doc_len": avg_doc_len
				}, 
				f,
				indent=4
			)
		
		# Update config too.
		with open("config.json", "r") as f_read:
			data = json.load(f_read)
		
		data["tf-idf_config"]["corpus_size"] = corpus_size
		data["bm25_config"]["corpus_size"] = corpus_size
		data["bm25_config"]["avg_doc_len"] = avg_doc_len

		with open("config.json", "w+") as f_write:
			json.dump(data, f_write, indent=4)
	
	print("All corpus level statistics have been calculated.")
	print(f"Results stored to {corpus_path}") 
	gc.collect()

	###################################################################
	# Stage 2: Compute Inverse Document Frequency (IDF)
	###################################################################
	print("Precomputing Inverse Document Frequencies and storing to staging...")

	# Path to output parquet.
	idf_path = os.path.join(idf_staging, "idf.parquet")

	# Load necessary corpus metadata (primarily want the corpus size
	# for this part).
	with open(corpus_path, "r") as f:
		corpus_data = json.load(f)
		corpus_size = corpus_data["corpus_size"]
		avg_doc_len = corpus_data["avg_doc_len"]

	# Perform processing if the end parquet file is not available.
	if not os.path.exists(idf_path):
		idf = compute_idf(word_to_docs_files, corpus_size, use_json)

		# Flatten data.
		idf_data = list()
		for word, idf_value in idf.items():
			idf_data.append((word, idf_value))

		# Convert to PyArrow Table. Table columns:
		# word (str),idf (float)
		table = pa.Table.from_pydict({
			"word": [record[0] for record in idf_data],
			"idf": [record[1] for record in idf_data],
		})

		# Save to Parquet file (store in staging).
		pq.write_table(table, idf_path)

	print("All Inverse Document Frequencies have been computed.")
	print(f"Results stored to {idf_path}") 
	gc.collect()

	###################################################################
	# Stage 3: Compute TF-IDF and BM25
	###################################################################
	print("Precomputing TF-IDF and BM25 values...")

	# Load the necessary data for filtering articles and computing the
	# TF-IDF and BM25 values.
	idf_df = pd.read_parquet(idf_path)
	with open(corpus_path, "r") as f:
		avg_doc_len = json.load(f)["avg_doc_len"]
	k1 = config["bm25_config"]["k1"]
	b = config["bm25_config"]["b"]

	# Iterate through each file.
	for idx, file in enumerate(doc_to_words_files):
		# Isolate the basename and print out the current file and
		# its position.
		basename = os.path.basename(file)
		print(f"Processing {basename} ({idx + 1}/{len(doc_to_words_files)})")

		# Path to output parquet.
		output_file = os.path.join(
			output_folder, 
			basename.replace(extension, ".parquet")
		)

		# Skipif the output parquet exists.
		if os.path.exists(output_file):
			continue

		# Initialize a pandas DataFrame object to hold the flattened 
		# data output.
		vector_data = pd.DataFrame()

		# Load the doc to words map.
		doc_to_words = load_data_file(file, use_json)

		# Remove the invalid articles from the keys (effectively ignore
		# invalid articles).
		valid_articles = list(doc_to_words.keys())

		# Chunk the data to enable concurrency/parallelism.
		chunk_size = math.ceil(len(valid_articles) / max_workers)
		articles_list = [
			valid_articles[i:i + chunk_size] 
			for i in range(0, len(valid_articles), chunk_size)
		]
		doc_to_words_list = [
			{
				article: doc_to_words[article] 
				for article in article_sublist
			}
			for article_sublist in articles_list
		]
		args_list = [
			(doc_to_words_subdict, idf_df, k1, b, avg_doc_len)
			for doc_to_words_subdict in doc_to_words_list
		]
			
		# Calculate each document/word's TF-IDF and BM25.
		if num_proc > 1:
			with mp.Pool(max_workers) as pool:
				results = pool.starmap(
					compute_sparse_vectors, args_list
				)
				for result in results:
					vector_data = pd.concat(
						[vector_data, result], ignore_index=True
					)
		else:
			with ThreadPoolExecutor(max_workers) as executor:
				results = executor.map(
					lambda args: compute_sparse_vectors(*args),
					args_list
				)
				for result in results:
					vector_data = pd.concat(
						[vector_data, result], ignore_index=True
					)

		# NOTE:
		# Dataframe contains the following features:
		# doc (str), word (str), tf (int), doc_len (int), idf (float),
		# tf_idf (float), bm25 (float)

		# Save to Parquet file (store in output).
		vector_data.to_parquet(output_file)

	# Clear loaded files.
	del idf_df

	print("All TF-IDF and BM25 values have been computed.")
	print(f"Results stored to {output_folder}") 
	gc.collect()

	###################################################################
	# Stage 4: Compute Inverted Index (Sharded Tries)
	###################################################################
	print("Precomputing Inverted Index Tries...")
	
	# Load vocabulary from IDF parquet.
	vocab = pd.read_parquet(idf_path)["word"].to_list()

	# Initialize dictionary from list of starting characters.
	starting_chars = list(string.ascii_lowercase) + ["other"]
	dictionary = {char: list() for char in starting_chars}

	# Load the document IDs from the parquets and compute the mappings
	# from each document and their unique ID.
	parquet_files = [
		os.path.join(output_folder, file)
		for file in os.listdir(output_folder)
		if file.endswith(".parquet")
	]

	# Path for doc to int mappings and their inverse.
	doc_to_int_path = os.path.join(
		trie_folder, f"doc_to_int{extension}"
	)
	int_to_doc_path = os.path.join(
		trie_folder, f"int_to_doc{extension}"
	)

	# Compute the mappings if they are missing.
	if not os.path.exists(doc_to_int_path) or not os.path.exists(int_to_doc_path):
		doc_to_int = dict()
		id_number = 1
		for file in tqdm(parquet_files, "Mapping documents to numbers"):
			documents = pd.read_parquet(file)["doc"].unique().tolist()
			for doc in documents:
				doc_to_int[doc] = id_number
				id_number += 1

		int_to_doc = {
			str(int_value): doc for doc, int_value in doc_to_int.items()
		}

		# Save those mappings.
		write_data_file(doc_to_int_path, doc_to_int, use_json)
		write_data_file(int_to_doc_path, int_to_doc, use_json)
	else:
		# Load the mappings.
		doc_to_int = load_data_file(doc_to_int_path, use_json)
		int_to_doc = load_data_file(int_to_doc_path, use_json)

	# Set chunk size for each starting character.
	trie_max_words = 500_000

	# Initialize metadata.
	trie_metadata = dict()
	max_word_len = 60

	# Variations of inverted index
	# Straight inverted index vs Trie based
	# - Trie offers better compression for storage at the cost of 
	#	compute time required for loading.
	# - Straight inverted index is easier to implement and 
	#	conceptualize.

	# Index sorted by most common words and chunked (ETA 84 hours).
	# Load the vocab and the document frequencies.
	vocab = dict()
	for word_to_docs_file in tqdm(word_to_docs_files, "Loading vocab"):
		vocab.update(load_data_file(word_to_docs_file, use_json))

	# Sort the words based on those with the most common occurences 
	# (highest document frequencies).
	# sorted_vocab = sorted(
	# 	vocab, key=lambda key: vocab[key], reverse=True # Work.
	# ) # For chunk by word
	sorted_vocab = sorted(
		[(word, count) for word, count in vocab.items()], 
		key=lambda x: x[1],
		reverse=True
	) # For chunk by number of documents
	
	# Graph highest number vocab values.
	# import matplotlib.pyplot as plt
	# vocab_values = sorted(list(vocab.values()), reverse=True)
	# for i in [500, 1000, 5000]:
	# 	vocab_slice = [
	# 		(idx + 1, value) 
	# 		for idx, value in enumerate(vocab_values[:i])
	# 	]
	# 	x = [vocab[0] for vocab in vocab_slice]
	# 	y = [vocab[1] for vocab in vocab_slice]

	# 	# Create the plot
	# 	plt.plot(x, y)

	# 	# Add labels and title (optional)
	# 	plt.xlabel('X-axis')
	# 	plt.ylabel('Y-axis')
	# 	plt.title(f'Vocab to Doc Freq (top-{i})')

	# 	plt.savefig(f'Vocab to Doc Freq (top-{i}).png')
	# exit()

	# NOTE:
	# The above code block that charts word document frequency (after 
	# sorting) gave interesting figures that empirically validated 
	# zipf's law. Therefore, it would make sense not to chunk based on
	# number of words but rather chunk based on the aggregation of the
	# number of documents each word is mapped to.
	
	# Adjust maximum number of aggregated documents according to memory
	# constraints. Number of files generated will fluctuated depending
	# on this value as well as the max_depth used.
	# max_aggr_doc_count = 1_000_000	# consumes ~ GB RAM
	max_aggr_doc_count = 500_000	# consumes ~ GB RAM
	# max_aggr_doc_count = 250_000
	# max_aggr_doc_count = 100_000

	# Chunk the sorted list based on max_aggr_doc_count.
	vocab_chunks = []
	current_chunk = []
	current_sum = 0

	for item in sorted_vocab:
		if current_sum + item[1] > max_aggr_doc_count:
			# Add the current chunk to the chunks list and start a new 
			# chunk.
			vocab_chunks.append(current_chunk)
			current_chunk = []
			current_sum = 0

		# current_chunk.append(item)
		current_chunk.append(item[0])
		current_sum += item[1]

	# Add the last chunk if it has any items
	if current_chunk:
		vocab_chunks.append(current_chunk)

	# Memory cleanup.
	del vocab
	del sorted_vocab
	gc.collect()

	for idx, chunk in enumerate(vocab_chunks):
		print(f"Processing vocab chunk {idx + 1}/{len(vocab_chunks)}")

		file = os.path.join(
			trie_folder, f"inverted_index_{idx + 1}{extension}"
		)

		if os.path.exists(file):
			continue

		word_doc_map = {word: list() for word in chunk}

		for parquet_file in tqdm(parquet_files, f"Scanning through parquets for chunk {idx + 1}"):
			df = pd.read_parquet(parquet_file)

			# Filter the dataframe for relevant words in the current chunk
			filtered_df = df[df["word"].isin(chunk)]

			# TODO:
			# Fix warning error that is given:
			# /home/diego/Documents/GitHub/WikipediaEnSearch/precompute_sparse_vectors.py:975: SettingWithCopyWarning: 
			# A value is trying to be set on a copy of a slice from a DataFrame.
			# Try using .loc[row_indexer,col_indexer] = value instead

			# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
			# filtered_df["doc"] = filtered_df["doc"].map(doc_to_int)

			# Map documents to their integer representation and group by 'word'
			filtered_df["doc"] = filtered_df["doc"].map(doc_to_int)
			grouped = filtered_df.groupby("word")["doc"].apply(set)

			# Update word_doc_map with the new documents
			for word, doc_set in grouped.items():
				word_doc_map[word].extend(doc_set - set(word_doc_map[word]))

		write_data_file(file, word_doc_map, use_json)


	print("All Inverted Index Tries have been computed.")
	print(f"Results stored to {trie_folder}") 
	gc.collect()

	# Clear staging (if applicable).
	if clear_staging in ["after", "both"]:
		clear_staging_folders(
			[idf_staging, corpus_staging, corpus_staging]
		)

	# Exit the program.
	exit(0)
	

if __name__ == '__main__':
	# Required to initialize models on GPU for multiprocessing. Placed
	# here due to recommendation from official python documentation.
	mp.set_start_method("spawn", force=True)
	main()