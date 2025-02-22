# search.py
# Implement search methods on the dowloaded (preprocessed) wikipedia
# data.
# Python 3.11
# Windows/MacOS/Linux


import concurrent.futures
import copy
import cProfile
import gc
import hashlib
import heapq
import json
import math
import mmap
import multiprocessing as mp
import os
import re
import string
import time
from typing import List, Dict, Tuple

from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import lancedb
import msgpack
import numpy as np
import pandas as pd
# import polars as pl
import pyarrow as pa
import torch
from tqdm import tqdm

from preprocess import load_model, process_page
from preprocess import bow_preprocessing, vector_preprocessing


profiler = cProfile.Profile()


def hashSum(data: str) -> str:
	'''
	Compute the SHA256SUM of the xml data. This is used as part of the
		naming scheme down the road.
	@param: data (str), the raw string data from the xml data.
	@return: returns the SHA256SUM hash.
	'''
	# Initialize the SHA256 hash object.
	sha256 = hashlib.sha256()

	# Update the hash object with the (xml) data.
	sha256.update(data.encode('utf-8'))

	# Return the digested hash object (string).
	return sha256.hexdigest()


def load_article_html_file(path: str) -> str:
	'''
	Load an html file from the given path.
	@param: path (str), the path of the html file that is to be loaded.
	@return: Returns the html file contents.
	'''
	with open(path, "r") as f:
		return f.read()


def load_article_text(path: str) -> str:
	'''
	Load the specified article from an html file given the path.
	@param: path (str), the path of the html file that is to be loaded.
	@return: Returns a string containing the article text of the file.
	'''
	# Validate the file path.
	if not os.path.exists(path):
		article = f"ERROR: COULD NOT LOCATE ARTICLE AT {path}"
		return article

	# Load the (html) file.
	file = load_article_html_file(path)

	# Parse the file with beautifulsoup.
	page_soup = BeautifulSoup(file, "lxml")

	# Isolate the article/page's raw text. Create copies for each
	# preprocessing task.
	try:
		article = process_page(page_soup)
	except:
		article = f"Unable to parse invalid article: {file}"

	# Return the list of article texts.
	return article


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


def create_aligned_tfidf_vector(group: pd.DataFrame | Dict[str, float], words: List[str]):
	'''
	Create a TF-IDF vector that aligns with the input words list.
	@param: group (), the grouped dataframe entries containing the 
		TF-IDF data for each word in the group (document).
	@param: words (List[str]), the input list of words.
	@return: returns the aligned TF-IDF vector as a List[float].
	'''
	if isinstance(group, pd.DataFrame):
		# Create a dictionary for quick lookup.
		tfidf_dict = dict(zip(group["word"], group["tf_idf"]))
	else:
		tfidf_dict = group

	# Align values with the order in 'words'.
	return [tfidf_dict.get(word, 0.0) for word in words]


def cosine_similarity(vec1: List[float], vec2: List[float]):
	'''
	Compute the cosine similarity of two vectors.
	@param: vec1 (List[float]), a vector of float values. This can be
		vectors such as a sparse vector from TF-IDF or BM25 to a dense
		vector like an embedding vector.
	@param: vec2 (List[float]), a vector of float values. This can be
		vectors such as a sparse vector from TF-IDF or BM25 to a dense
		vector like an embedding vector.
	@return: Returns the cosine similarity between the two input 
		vectors. Value range is 0 (similar) to 1 (disimilar).
	'''
	# Convert the vectors to numpy arrays.
	np_vec1 = np.array(vec1)
	np_vec2 = np.array(vec2)

	# Compute the cosine similarity of the two vectors and return the
	# value.
	cosine = np.dot(np_vec1, np_vec2) /\
		(np.linalg.norm(np_vec1) * np.linalg.norm(np_vec2))
	return cosine
	

def print_results(results: List, search_type: str = "tf-idf", print_doc: bool = False) -> None:
	valid_search_types = ["tf-idf", "bm25", "vector", "rerank"]
	assert search_type in valid_search_types,\
		f"Expected 'search_type' to be either {', '.join(valid_search_types)}. Received {search_type}"

	# Format of the input search results:
	# TF-IDF
	# [cosine similarity, document (file), text, indices]
	# BM25
	# [score, document (file), text, indices]
	# Vector
	# [cosine similarity, document (file), text, indices]
	# ReRank
	# [cosine similarity, document (file), text, indices]

	print(f"SEARCH RESULTS:")
	print('-' * 72)
	for result in results:
		# Deconstruct the results.
		score, document, text, indices = result

		# Print the results out.
		print(f"Score: {score}")
		print(f"File Path: {document}")
		if print_doc:
			print(f"Article Text:\n{text[indices[0]:indices[1]]}")


class InvertedIndex:
	def __init__(self, index_dir: str, use_json: bool = False, use_multiprocessing: bool = False) -> None:
		extension = ".json" if use_json else ".msgpack"
		self.use_json = use_json
		self.use_multiprocessing = use_multiprocessing

		# Isolate map for int to doc (load as well).
		self.int_to_doc_file = f"int_to_doc{extension}"
		self.int_to_doc_path = os.path.join(
			index_dir, self.int_to_doc_file
		)
		self.int_to_doc = load_data_file(
			self.int_to_doc_path, use_json
		)

		self.index_files = [
			os.path.join(index_dir, file) 
			for file in os.listdir(index_dir)
			if file.endswith(extension) and "_to_" not in file
		]
		self.index_files = [
			file for file in self.index_files
			if os.path.isfile(file)
		] # Filter to only contain files.


	def query(self, words: List[str], num_workers: int = 8) -> List[str]:
		chunk_size = math.ceil(len(self.index_files) / num_workers)
		chunks = [
			self.index_files[i:i + chunk_size]
			for i in range(0, len(self.index_files), chunk_size)
		]
		args_list = [(words, index_files) for index_files in chunks]
		docs_set = set()

		if self.use_multiprocessing:
			num_workers = min(mp.cpu_count(), num_workers)
			with mp.Pool(num_workers) as pool:
				results = pool.starmap(
					self.get_docs_from_file, args_list
				)
				for result in results:
					docs_set.update(result)
		else:
			with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
				results = executor.map(
					lambda args: self.get_docs_from_file(*args),
					args_list
				)
				for result in results:
					docs_set.update(result)

		return list(docs_set)


	def get_docs_from_file(self, words: List[str], files: List[str]) -> List[str]:
		docs_set = set()
		for file in tqdm(files):
			with open(file, "rb") as f:
				# Memory-map the file
				mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)

				if self.use_json:
					# Load JSON content incrementally
					data = json.loads(mm.read().decode('utf-8'))
				else:
					# Decode Msgpack content incrementally
					unpacker = msgpack.Unpacker(mm, raw=False)
					data = unpacker.unpack()
				
				for word in words:
					if word in list(data.keys()):
						docs_set.update(data[word])
				
				# Clean up
				mm.close()
		
		return self.decode_documents(list(docs_set))


	def decode_documents(self, document_ids: List[str]) -> List[str]:
		return [self.int_to_doc[str(doc)] for doc in document_ids]
	

class SortedInvertedIndex(InvertedIndex):
	def __init__(self, index_dir: str, use_json: bool = False, use_multiprocessing: bool = False) -> None:
		extension = ".json" if use_json else ".msgpack"
		self.extension = extension
		self.use_json = use_json
		self.use_multiprocessing = use_multiprocessing

		# Isolate map for int to doc (load as well).
		self.int_to_doc_file = f"int_to_doc{extension}"
		self.int_to_doc_path = os.path.join(
			index_dir, self.int_to_doc_file
		)
		self.int_to_doc = load_data_file(
			self.int_to_doc_path, use_json
		)

		self.index_files = [
			os.path.join(index_dir, file) 
			for file in os.listdir(index_dir)
			if file.endswith(extension) and "_to_" not in file
		]
		self.index_files = [
			file for file in self.index_files
			if os.path.isfile(file)
		] # Filter to only contain files.
		self.index_files = sorted(
			self.index_files, key=self.get_number
		) # Sort by the file number (NOT lexicographical).


	def query(self, words: List[str]) -> List[str]:
		# Initialize the set to contain the list of unique document 
		# IDs.
		docs_set = set()
		words_list = copy.deepcopy(words)

		# Iterate through the inverted index files.
		for file in tqdm(self.index_files):
			# Skip remaining files if the query words list is empty.
			if len(words) == 0:
				continue

			# Initialize a list/set to keep track of words found so 
			# far.
			found_words = set()

			with open(file, "rb") as f:
				# Memory-map the file.
				mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)

				if self.use_json:
					# Load JSON content incrementally
					data = json.loads(mm.read().decode('utf-8'))
				else:
					# Decode Msgpack content incrementally
					unpacker = msgpack.Unpacker(mm, raw=False)
					data = unpacker.unpack()
				
				# Iterate through all query words and search for each
				# key/value pair for the file.
				for word in words_list:
					if word in list(data.keys()):
						docs_set.update(data[word])
						found_words.add(word)
				
				# Clean up
				mm.close()

			# Remove found words from the query list.
			for word in found_words:
				words_list.remove(word)

			# Memory cleanup.
			del found_words
			gc.collect()
		
		# Return the decoded document IDs list (as document paths).
		return self.decode_documents(list(docs_set))
	

	def get_number(self, filename: str) -> int:
		match = re.search(
			r"inverted_index_(\d+)" + self.extension, filename
		)
		return int(match.group(1)) if match else -1  # Default to -1 if no match


	def decode_documents(self, document_ids: List[str]) -> List[str]:
		return [self.int_to_doc[str(doc)] for doc in document_ids]


class BagOfWords: 
	def __init__(self, bow_dir: str, depth: int = 1, corpus_size: int=-1, srt: float=-1.0, use_json: bool = False) -> None:
		'''
		Initialize a Bag-of-Words search object.
		@param: bow_dir (str), a path to the directory containing the 
			bag of words metadata. This metadata includes the folders
			mapping to the word-to-document and document-to-word files.
		@param: srt (float), the sparse retrieval threshold. A value
			used to remove documents from the search results if they
			score a cosine similarity above the threshold.
		'''
		# Initialize class variables either from arguments or with
		# default values.

		# Metadata file paths.
		self.bow_dir = bow_dir
		self.term_article_graph_path = None
		self.trie_folder = None
		self.inverted_index_files = None
		self.sparse_vector_files = None
		self.int_to_doc_file = None
		self.documents_folder = "./InvestopediaDownload/data"

		# Corpus metadata.
		self.corpus_size = corpus_size	# total number of documents (articles)
		self.srt = srt					# similarity relative threshold value
		
		# File path extensions.
		self.use_json = use_json
		self.extension = ".json" if use_json else ".msgpack"
		
		# Additional variables.
		self.alpha_numerics = string.digits + string.ascii_lowercase
		self.word_len_limit = 60
		self.depth = depth

		# Initialize mapping folder path and files list.
		self.locate_and_validate_documents(bow_dir, depth)

		# Verify that the class variables that were initialized as None
		# by default are no longer None.
		initialized_variables = [
			self.sparse_vector_files, self.trie_folder, 
			self.inverted_index_files, self.int_to_doc_file,
		]
		assert None not in initialized_variables,\
			"Some variables were not initialized properly"
		
		# Initialize inverted index.
		self.inverted_index = SortedInvertedIndex(
			self.trie_folder, self.use_json
		)

		# Compute the corpus size if the default value for the argument 
		# is detected.
		if self.corpus_size == -1:
			self.corpus_size = self.get_number_of_documents()

		# Verify that the corpus size is not 0.
		assert self.corpus_size != 0,\
			"Could not count the number of documents (articles) in the corpus. Corpus size is 0."
		
		# Verify that the srt is either -1.0 or in the range [0.0, 1.0]
		# (cosine similarity range).
		srt_off = self.srt == -1.0
		srt_valid = self.srt >= 0.0 and self.srt <= 1.0
		assert srt_off or srt_valid,\
			"SRT value was initialize to an invalid number. Either -1.0 for 'off' or a float in the range [0.0, 1.0] is expected"


	def locate_and_validate_documents(self, bow_dir: str, depth: int):
		'''
		Verify that the bag-of-words directory exists along with the
			metadata files expected to be within them.
		@param: bow_dir (str), a path to the directory containing the 
			bag of words metadata. This metadata includes the folders
			mapping to the word-to-document and document-to-word files.
		@return: returns nothing.
		'''
		# Initialize path to word to inverted index and sparse vectors 
		# folders (inverted index folder contains document id to 
		# document mapping as well).
		self.trie_folder = os.path.join(
			bow_dir, "tries", f"depth_{depth}"
		)
		self.sparse_vectors_folder = os.path.join(
			bow_dir, "sparse_vectors", f"depth_{depth}"
		)

		# Isolate all files for each folder.
		self.inverted_index_files = [
			os.path.join(self.trie_folder, file)
			for file in os.listdir(self.trie_folder)
			if file.endswith(self.extension)
		]
		self.sparse_vector_files = [
			os.path.join(self.sparse_vectors_folder, file)
			for file in os.listdir(self.sparse_vectors_folder)
			if file.endswith(".parquet")
		]

		# Initialize path to word to idf mappings, trie files, and 
		# document id to document mappings.
		doc_id_map_files = [
			"doc_to_int" + self.extension,
			"int_to_doc" + self.extension,
		]
		self.int_to_doc_file = os.path.join(
			self.trie_folder, doc_id_map_files[1]
		)

		# Initialize path to term-article JSON.
		base = f"term_article_graph_depth{depth}.json" if depth > 1 else ""
		self.term_article_graph_path = os.path.join(
			"./InvestopediaDownload/graph", base 
		) if depth > 1 else ""
		
		# Verify that the list of files for each mapping folder is not
		# empty.
		assert os.path.exists(self.int_to_doc_file),\
			f"Required document id to document file (int_to_doc{self.extension}) in {self.trie_folder} does exist"
		assert len(self.inverted_index_files) != 0,\
			f"Required inverted index to be initialized in {self.trie_folder}. Found no files."
		assert len(self.sparse_vector_files) != 0,\
			f"Required sparse vector data to be initialized in {self.sparse_vectors_folder}. Found no files."


	def get_number_of_documents(self) -> int:
		'''
		Count the number of documents recorded in the corpus.
		@param, takes no arguments.
		@return, Returns the number of documents in the corpus.
		'''
		# Initialize the counter to 0.
		counter = 0

		# Iterate through each file in the documents to words map 
		# files.
		print("Getting the number of documents in the corpus...")
		for file in tqdm(self.sparse_vector_files):
			# Load the data from the file and increment the counter by
			# the number of documents in each file.
			df = pd.read_parquet(file)
			counter += df["doc"].nunique()
		
		# Return the count.
		return counter


	def get_document_paths_from_documents(self, documents: List[str]) -> Dict[str, List[str]]:
		'''
		Given a document list (each document is a file + article hash),
			return a dictionary mapping the full path of all unique 
			files to the expected article hashes within each file.
		@param: documents (List[str]), the list of all documents that
			were returned by the inverted index.
		@return: returns a dictionary mapping all files to the 
			respective article hashes within each file.
		'''
		# Initialize dictionary mapping each file to the list of 
		# expected hashes.
		file_article_dict = dict()

		# Isolate directory path and basename of file.
		basenames = [
			(os.path.dirname(doc), os.path.basename(doc))
			for doc in documents
		]

		# for folder, name in basenames:
		for _, name in basenames:
			file_basename, article_hash = name.split(".xml")
			file_basename += ".xml"
			# file = os.path.join(folder, file_basename)
			file = file_basename

			if file in list(file_article_dict.keys()):
				file_article_dict[file].append(article_hash)
			else:
				file_article_dict[file] = [article_hash]

		return file_article_dict
	

	def compute_tf(self, doc_word_freq: Dict, words: List[str]) -> List[float]:
		'''
		Compute the Term Frequency of a set of words given a document's
			word frequency mapping.
		@param: doc_word_freq (Dict), the mapping of a given word the
			frequency it appears in a given document.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to compute the Inverse Document Frequency for.
		@return: returns the Term Frequency of each of the words for
			the given document in a vector (List[float]). The vector is
			ordered such that the index of each value corresponds to 
			the index of a word in the word list argument.
		'''
		# Initialize the document's term frequency vector.
		doc_word_tf = [0.0] * len(words)

		# Compute total word count.
		total_word_count = sum(
			[value for value in doc_word_freq.values()]
		)

		# Compute the term frequency accordingly and add it to the 
		# document's word vector
		for word_idx in range(len(words)):
			word = words[word_idx]
			if word in doc_word_freq:
				word_freq = doc_word_freq[word]
				doc_word_tf[word_idx] =  word_freq / total_word_count
			
		# Return the document's term frequency for input words as a 
		# vector (List[float]).
		return doc_word_tf
	

	def compute_idf(self, words: List[str], return_dict: bool = False) -> List[float] | Dict[str, float]:
		'''
		Retrieve the precomputed Inverse Document Frquency of the given
		 	set of (usually query) words.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to compute the Inverse Document Frequency for.
		@param: return_dict (bool), whether to retunr the precomputed 
			IDF as an ordered list or dictionary. Default is False.
		@param: returns the Inverse Document Frequency for all words
			queried in the corpus. The data is returned in an ordered
			list (List[float]) where the index of each value
			corresponds to the index of a word in the word list 
			argument OR a dictionary (Dict[str, float]).
		'''
		# Initialize a list containing the mappings of the query words
		# to the IDF.
		idf_vector = [0.0] * len(words)
		idf_dict = dict()

		# Iterate through each file.
		# for file in self.idf_files:
		# 	# Load the word to IDF mappings from file.
		# 	word_to_idf = load_data_file(file, use_json=self.use_json)

		# 	# Iterate through each word and retrieve the IDF value for
		# 	# that word if it is available.
		# 	for word_idx in range(len(words)):
		# 		word = words[word_idx]
		# 		if word in word_to_idf:
		# 			idf_vector[word_idx] = word_to_idf[word]
		# 			idf_dict[word] = word_to_idf[word]

		for file in self.sparse_vector_files:
			if all(word in (idf_dict.keys()) for word in words):
				continue

			df = pd.read_parquet(file)

			# Extract the unique word-IDF pairs.
			word_idf_map = df[['word', 'idf']].drop_duplicates(subset='word')\
				.set_index('word')['idf']\
				.to_dict()

			# Get the IDF values for words in vocab, with default value if a word is not in the DataFrame
			# idf_values = {word: word_idf_map.get(word, None) for word in words}
			idf_values = {
				word: idf 
				for word in words 
				if (idf := word_idf_map.get(word)) is not None
			}
			idf_dict.update(idf_values)

		idf_vector = [
			idf_dict[word]
			for word in words
		]

		# Return the inverse document frequency vector.
		return idf_vector if not return_dict else idf_dict


class TF_IDF(BagOfWords):
	def __init__(self, bow_dir: str, depth: int = 1, corpus_size: int=-1, srt: float=-1.0, use_json=False) -> None:
		super().__init__(
			bow_dir=bow_dir, depth=depth, corpus_size=corpus_size, 
			srt=srt, use_json=use_json
		)


	def search(self, query: str, max_results: int = 50):
		'''
		Conducts a search on the wikipedia data with TF-IDF.
		@param: query str, the raw text that is being queried from the
			wikipedia data.
		@param: max_results (int), the maximum number of search results
			to return.
		@return: returns a list of objects where each object contains
			an article's path, title, retrieved text, and the slices of
			that text that is being returned (for BM25 and TF-IDF, 
			those slices values are for the whole article).
		'''
		# Assertion for max_results argument (must be non-zero int).
		assert isinstance(max_results, int) and str(max_results).isdigit() and max_results > 0, f"max_results argument is expected to be some int value greater than zero. Recieved {max_results}"

		# Preprocess the search query to a bag of words.
		words, word_freq = bow_preprocessing(query, True)

		# Sort the words (as part of vectorization).
		words = sorted(words)

		# Query inverted index.
		document_ids = self.inverted_index.query(words)

		# Get word IDF for query terms.
		word_idfs = self.compute_idf(words)

		# Query TF-IDF.
		query_tfidf = {
			word: word_freq[word] * word_idfs[idx]
			for idx, word in enumerate(words)
		}
		query_tfidf_vector = create_aligned_tfidf_vector(
			query_tfidf, words
		)

		# TODO:
		# Add num_workers as an argument to either __init__() or here 
		# for the BagOfWords class and set it in the config.json.

		num_workers = 8
		num_workers = min(num_workers, len(self.sparse_vector_files))
		chunk_size = math.ceil(
			len(self.sparse_vector_files) / num_workers
		)
		file_chunks = [
			self.sparse_vector_files[i:i + chunk_size]
			for i in range(0, len(self.sparse_vector_files), chunk_size)
		]
		args_list = [
			(document_ids, file_chunk, words, query_tfidf_vector, max_results)
			for file_chunk in file_chunks
		]
		corpus_tfidf = []

		# Snippet. Polars does not play well with multithreading from
		# python because it is already multithreading under the hood in
		# rust.
		
		# Use with Pandas/all other software.
		with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
			print("Starting multithreaded processing")
			results = executor.map(
				lambda args: self.targeted_file_search(*args), 
				args_list
			)
			
			for result in results:
				while len(result) > 0:
					result_item = result.pop()
					if result_item in corpus_tfidf:
						continue
					if max_results != -1 and len(corpus_tfidf) >= max_results:
						# Pushpop the highest (cosine similarity) value
						# tuple from the heap to make way for the next
						# tuple.
						heapq.heappushpop(
							corpus_tfidf,
							result_item
						)
					else:
						heapq.heappush(
							corpus_tfidf,
							result_item
						)

		# The corpus TF-IDF results are stored in a max heap. Convert
		# the structure back to a list sorted from smallest to largest
		# cosine similarity score.
		sorted_rankings = []
		for _ in range(len(corpus_tfidf)):
			# Pop the top item from the max heap.
			result = heapq.heappop(corpus_tfidf)

			# Reverse the cosine similarity score back to its original
			# value.
			result[0] *= -1

			# Extract the document path load the article text.
			document_path = result[1]
			text = load_article_text(document_path)
			
			# Append the results.
			full_result = result + [text, [0, len(text)]]

			# Insert the item into the list from the back.
			sorted_rankings.insert(-1, full_result)

		# Return the list.
		return sorted_rankings
	

	def targeted_file_search(self, document_ids: List[str], sparse_vector_files: List[str], words: List[str], query_tfidf_vector: List[float], max_results: int):
		# Stack heap for the search.
		stack_heap = list()
		heapq.heapify(stack_heap)

		for file in tqdm(sparse_vector_files):
			# profiler.enable()

			###########################################################
			# PANDAS
			###########################################################
			# Read in the doc_to_words data into a dataframe.
			file = file.replace(".msgpack", ".parquet")
			df_doc2words = pd.read_parquet(file)
			target_docs = document_ids

			# Filter out all entries that are not within the specified 
			# document IDs (no need to worry about redirect articles).
			# df_doc2words = df_doc2words[df_doc2words["doc"].isin(target_docs)]

			# Filter out all entries without the words from the input 
			# list.
			# df_doc2words = df_doc2words[df_doc2words["word"].isin(words)]

			# Convert doc and word entries to str (they're currently 
			# stored as object dtypes).
			# df_doc2words["doc"] = df_doc2words["doc"].astype(str)
			# df_doc2words["word"] = df_doc2words["word"].astype(str)
			df_doc2words["doc"] = df_doc2words["doc"].apply(str)
			df_doc2words["word"] = df_doc2words["word"].apply(str)

			# Isolate entries where the document IDs are within the set
			# of target documents and the words for those documents and
			# within the set of target words.
			df_doc2words = df_doc2words[
				df_doc2words["word"].isin(words) & df_doc2words["doc"].isin(target_docs)
			]

			# Group tf-idf values by document to get a document level
			# tf-idf vector.
			doc_vectors = df_doc2words.groupby("doc")\
				.apply(lambda group: create_aligned_tfidf_vector(group, words))
			
			# Compute the document cosine similarity.
			results = doc_vectors.reset_index(name="tfidf_vector")
			results["cosine_similarity"] = results["tfidf_vector"].apply(
				lambda vec: cosine_similarity(vec, query_tfidf_vector)
			)

			# Grab the top n results and store it to a heap.
			top_n = results.nlargest(max_results, "cosine_similarity")
			top_n_list = []
			for _, row in top_n.iterrows():
				top_n_list.append((-row["cosine_similarity"], row["doc"]))

			for doc_cos_score, doc in top_n_list:
				# Insert the document name (includes file path & 
				# article SHA1), TF-IDF vector, and cosine similarity 
				# score (against the query TF-IDF vector) to the heapq.
				# The heapq sorts by the first value in the tuple so 
				# that is why the cosine similarity score is the first
				# item in the tuple.
				if max_results != -1 and len(stack_heap) >= max_results:
					# Pushpop the highest (cosine similarity) value
					# tuple from the heap to make way for the next
					# tuple.
					heapq.heappushpop(stack_heap, [doc_cos_score, doc])
				else:
					# Push the highest (cosine similarity) value tuple 
					# from the heap to make way for the next tuple.
					heapq.heappush(stack_heap, [doc_cos_score, doc])

			# profiler.disable()
			# profiler.print_stats(sort="time")

		return stack_heap
	

	def compute_tfidf(self, words: List[str], query_word_freq: Dict, documents: List[str], max_results: int = -1):
		'''
		Iterate through all the documents in the corpus and compute the
			TF-IDF for each document in the corpus. Sort the results
			based on the cosine similarity score and return the sorted
			list.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to compute the Inverse Document Frequency for.
		@param: query_word_freq (Dict), the word frequency mapping for 
			the input search query.
		@param: documents (List[str]), the list of files/documents that
			will be queried from the corpus.
		@param: max_results (int), the maximum number of results to 
			return. Default is -1 (no limit).
		@return: returns the sorted list of search results. 
		'''
		# Sort the set of words (ensures consistent positions of each 
		# word in vector).
		words = sorted(words)

		# Iterate through all files to get IDF of each query word. 
		# Compute only once per search.
		word_idf = self.compute_idf(words)

		# Compute query TF-IDF.
		query_total_word_count = sum(
			[value for value in query_word_freq.values()]
		)
		query_tfidf = [0.0] * len(words)
		for word_idx in range(len(words)):
			word = words[word_idx]
			query_word_tf = query_word_freq[word] / query_total_word_count
			query_tfidf[word_idx] = query_word_tf * word_idf[word_idx]

		# Compute corpus TF-IDF.
		corpus_tfidf_heap = []

		# Given the documents, get a filtered list of documents to use 
		# from doc_to_word_files.
		file_to_article = self.get_document_paths_from_documents(
			documents
		)
		filtered_files = [
			file 
			for file in self.doc_to_word_files
			if os.path.basename(file).replace(self.extension, ".xml") in list(file_to_article.keys())
		]
		# print("File to articles")
		# print(json.dumps(file_to_article, indent=4))
		# print(f"Matching files")
		# print(json.dumps(filtered_files, indent=4))

		# NOTE:
		# Heapq in use is a max-heap. This is implemented by 
		# multiplying the cosine similarity score by -1. That way, the
		# largest values are actually the smallest in the heap and are
		# popped when we need to pushpop the largest scoring tuple.
		print("Running TF-IDF search...")

		# Compute TF-IDF for every file.
		# for file in tqdm(self.doc_to_word_files):
		for file in tqdm(filtered_files):
			# Load the doc to word frequency mappings from file.
			doc_to_words = load_data_file(file, self.use_json)

			# Compute the intersection of the documents passed in from
			# arguments and the current list of documents in the file.
			document_intersect = set(documents).intersection(
				list(doc_to_words.keys())
			)

			# Iterate through each document.
			# for doc in doc_to_words:
			# Iterate through the documents provided from arguments.
			# for doc in documents:
			# Iterate through the document intersection.
			for doc in list(document_intersect):
				# Extract the document word frequencies.
				word_freq_map = doc_to_words[doc]

				# Compute the document's term frequency for each word.
				doc_word_tf = self.compute_tf(word_freq_map, words)

				# Compute document TF-IDF.
				doc_tfidf = [
					tf * idf 
					for tf, idf in list(zip(doc_word_tf, word_idf))
				]

				# Compute cosine similarity against query TF-IDF and
				# the document TF-IDF.
				doc_cos_score = cosine_similarity(
					query_tfidf, doc_tfidf
				)

				# If the sparse retrieval threshold has been 
				# initialized, verify the document cosine similarity
				# score is within that threshold. Do not append
				# documents to the results list if they fall under the
				# threshold.
				if self.srt > 0.0 and doc_cos_score > self.srt:
					continue

				# Multiply score by -1 to get inverse score. This is
				# important since we are relying on a max heap.
				doc_cos_score *= -1

				# NOTE:
				# Using heapq vs list keeps sorting costs down: 
				# list sort is n log n
				# list append is 1 or n (depending on if the list needs
				# to be resized)
				# heapify is n log n but since heap is initialized
				# from empty list, that cost is negligible
				# heapq pushpop is log n
				# heapq push is log n
				# heapq pop is log n
				# If shortening the list is a requirement, then I 
				# dont have to worry about sorting the list before
				# slicing it with heapq. The heapq will maintain
				# order with each operation at a cost efficent speed.
				
				# Insert the document name (includes file path & 
				# article SHA1), TF-IDF vector, and cosine similarity 
				# score (against the query TF-IDF vector) to the heapq.
				# The heapq sorts by the first value in the tuple so 
				# that is why the cosine similarity score is the first
				# item in the tuple.
				if max_results != -1 and len(corpus_tfidf_heap) >= max_results:
					# Pushpop the highest (cosine similarity) value
					# tuple from the heap to make way for the next
					# tuple.
					heapq.heappushpop(
						corpus_tfidf_heap,
						# tuple([doc_cos_score, doc, doc_tfidf])
						# [doc_cos_score, doc, doc_tfidf]
						[doc_cos_score, doc]
					)
				else:
					heapq.heappush(
						corpus_tfidf_heap,
						# tuple([doc_cos_score, doc, doc_tfidf]) # Tuple doesnt support modification
						# [doc_cos_score, doc, doc_tfidf] # Results in issues unpacking list in preint_results()
						[doc_cos_score, doc]
					)

		# Return the corpus TF-IDF.
		return corpus_tfidf_heap


class BM25(BagOfWords):
	def __init__(self, bow_dir: str, depth: int = 1, k1: float = 1.0, b: float = 0.0, 
			  	corpus_size: int=-1, avg_doc_len: float=-1.0, srt: 
				float=-1.0, use_json=False) -> None:
		super().__init__(
			bow_dir=bow_dir, depth=depth, corpus_size=corpus_size, 
			srt=srt, use_json=use_json
		)
		self.avg_corpus_len = avg_doc_len
		if self.avg_corpus_len < 0.0:
			self.avg_corpus_len = self.compute_avg_corpus_size()
		self.k1 = k1
		self.b = b


	def compute_avg_corpus_size(self) -> float:
		'''
		Compute the average document length (in words) of the corpus.
		@param: takes no arguments
		@return: returns the average number of words per document 
			across the entire corpus.
		'''
		# Initialize document size sum.
		doc_size_sum = 0

		# Iterate through each file in the documents to words map 
		# files.
		print("Computing average document length of corpus...")
		for file in tqdm(self.sparse_vector_files):
			# Load the data from the file.
			df = pd.read_parquet(file)

			# Size of a document is equal to the sum of all term
			# frequencies for all words in the document. The sum of all
			# document sizes is the sum of that value (document size) 
			# for all documents in the corpus.
			doc_size_sum += df["tf"].sum()

		# Return the average document size.
		return doc_size_sum / self.corpus_size


	def search(self, query: str, max_results: int = 50):
		'''
		Conducts a search on the wikipedia data with BM25.
		@param: query str, the raw text that is being queried from the
			wikipedia data.
		@param: max_results (int), the maximum number of search results
			to return.
		@return: returns a list of objects where each object contains
			an article's path, title, retrieved text, and the slices of
			that text that is being returned (for BM25 and TF-IDF, 
			those slices values are for the whole article).
		'''
		# Assertion for max_results argument (must be non-zero int).
		assert isinstance(max_results, int) and str(max_results).isdigit() and max_results > 0, f"max_results argument is expected to be some int value greater than zero. Recieved {max_results}"

		# Preprocess the search query to a bag of words.
		words = bow_preprocessing(query, False)
		words = words[0] # unpack return tuple.

		# NOTE:
		# No need to sort words as part of vectorization because BM25
		# works on taking a score sum rather than relying on vectors.

		# Query inverted index.
		document_ids = self.inverted_index.query(words)

		num_workers = 8
		num_workers = min(num_workers, len(self.sparse_vector_files))
		chunk_size = math.ceil(
			len(self.sparse_vector_files) / num_workers
		)
		file_chunks = [
			self.sparse_vector_files[i:i + chunk_size]
			for i in range(0, len(self.sparse_vector_files), chunk_size)
		]
		args_list = [
			(document_ids, file_chunk, words, max_results)
			for file_chunk in file_chunks
		]
		corpus_bm25 = []

		# Use with Pandas/all other software.
		with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
			print("Starting multithreaded processing")
			results = executor.map(
				lambda args: self.targeted_file_search(*args), 
				args_list
			)
			
			for result in results:
				while len(result) > 0:
					result_item = result.pop()
					if result_item in corpus_bm25:
						continue
					if max_results != -1 and len(corpus_bm25) >= max_results:
						# Pushpop the highest (cosine similarity) value
						# tuple from the heap to make way for the next
						# tuple.
						heapq.heappushpop(
							corpus_bm25,
							result_item
						)
					else:
						heapq.heappush(
							corpus_bm25,
							result_item
						)

		# Compute the BM25 for the corpus.
		# corpus_bm25 = self.compute_bm25(
		# 	words, documents_ids, max_results=max_results
		# )

		# The corpus BM25 results are stored in a max heap. Convert the
		# structure back to a list sorted from largest to smallest BM25
		# score.
		sorted_rankings = []
		for _ in range(len(corpus_bm25)):
			# Pop the top item from the max heap.
			result = heapq.heappop(corpus_bm25)

			# Reverse the cosine similarity score back to its original
			# value.
			result[0] *= -1

			# Extract the document path load the article text.
			document_path = result[1]
			text = load_article_text(document_path)
			
			# Append the results.
			full_result = result + [text, [0, len(text)]]

			# Insert the item into the list from the back.
			sorted_rankings.insert(-1, full_result)

		# Return the list.
		return sorted_rankings


	def compute_bm25(self, words: List[str], documents: List[str], max_results: int = -1):
		'''
		Iterate through all the documents in the corpus and compute the
			BM25 for each document in the corpus. Sort the results
			based on the cosine similarity score and return the sorted
			list.
		@param: words (List[str]), the (ordered) list of all (unique) 
			terms to compute the Inverse Document Frequency for.
		@param: query_word_freq (Dict), the word frequency mapping for 
			the input search query.
		@param: documents (List[str]), the list of files/documents that
			will be queried from the corpus.
		@param: max_results (int), the maximum number of results to 
			return. Default is -1 (no limit).
		@return: returns the BM25 for the query as well as the sorted
			list of search results. 
		'''
		# Sort the set of words (ensures consistent positions of each 
		# word in vector).
		words = sorted(words)

		# Iterate through all files to get IDF of each query word. 
		# Compute only once per search.
		word_idf = self.compute_idf(words)

		# Compute corpus BM25.
		corpus_bm25_heap = []

		# Given the documents, get a filtered list of documents to use 
		# from doc_to_word_files.
		file_to_article = self.get_document_paths_from_documents(
			documents
		)
		filtered_files = [
			file 
			for file in self.doc_to_word_files
			if os.path.basename(file).replace(self.extension, ".xml") in list(file_to_article.keys())
		]

		# NOTE:
		# Heapq in use is a max-heap. In this case, we don't want to 
		# multiply the BM25 score by -1 because a larger score means a
		# document is "more relevant" to the query (so we want to drop
		# the lower scores if we have a max_results limit). BM25 also 
		# doesn't require using cosine similarity since it aggregates
		# the term values into a sum for the document score.
		print("Running BM25 search...")

		# Compute BM25 for every file.
		# for file in tqdm(self.doc_to_word_files):
		for file in tqdm(filtered_files):
			# Load the doc to word frequency mappings from file.
			doc_to_words = load_data_file(file, self.use_json)

			document_intersect = set(documents).intersection(
				list(doc_to_words.keys())
			)

			# Iterate through each document.
			# for doc in doc_to_words:
			# for doc in documents:
			for doc in document_intersect:
				# Initialize the BM25 score for the document.
				bm25_score = 0.0

				# Extract the document word frequencies.
				word_freq_map = doc_to_words[doc]

				# Compute the document length.
				doc_len = sum(
					[value for value in word_freq_map.values()]
				)

				# Compute the document's term frequency for each word.
				doc_word_tf = self.compute_tf(word_freq_map, words)

				# Iterate over the different words and compute the BM25
				# score for each. Aggregate that score by adding it to 
				# the total BM25 score value.
				for word_idx in range(len(words)):
					tf = doc_word_tf[word_idx]
					numerator = word_idf[word_idx] * tf * (self.k1 + 1)
					denominator = tf + self.k1 *\
						(
							1 - self.b + self.b *\
							(doc_len / self.avg_corpus_len)
						)
					bm25_score += numerator / denominator

				# NOTE:
				# We ignore similarity relevance threshold here because
				# the range of values for BM25 scores are outside of
				# the range of values we've set for srt [0.0, 1.0].
				# Makes it a headache to deal with an unbounded range
				# so we'll make do without this optimization.

				# Insert the document name (includes file path & 
				# article SHA1), BM25 score to the heapq. The heapq 
				# sorts by the first value in the tuple so that is why
				# the cosine similarity score is the first item in the 
				# tuple.
				if max_results != -1 and len(corpus_bm25_heap) >= max_results:
					# Pushpop the smallest (BM25) value tuple from the 
					# heap to make way for the next tuple.
					heapq.heappushpop(
						corpus_bm25_heap,
						# tuple([bm25_score, doc])
						[bm25_score, doc]
					)
				else:
					heapq.heappush(
						corpus_bm25_heap,
						# tuple([bm25_score, doc])
						[bm25_score, doc]
					)

		# Return the corpus BM25 rankings.
		return corpus_bm25_heap
	

	def targeted_file_search(self, document_ids: List[str], sparse_vector_files: List[str], words: List[str], max_results: int):
		# Stack heap for the search.
		stack_heap = list()
		heapq.heapify(stack_heap)

		for file in tqdm(sparse_vector_files):
			# Read in the doc_to_words data into a dataframe.
			file = file.replace(".msgpack", ".parquet")
			df_doc2words = pd.read_parquet(file)
			target_docs = document_ids

			# Convert doc and word entries to str (they're currently 
			# stored as object dtypes).
			df_doc2words["doc"] = df_doc2words["doc"].apply(str)
			df_doc2words["word"] = df_doc2words["word"].apply(str)

			# Isolate entries where the document IDs are within the set
			# of target documents and the words for those documents and
			# within the set of target words.
			df_doc2words = df_doc2words[
				df_doc2words["word"].isin(words) & df_doc2words["doc"].isin(target_docs)
			]

			# Group bm25 values by document to get a document level 
			# bm25 value.
			doc_vectors = df_doc2words.groupby("doc")["bm25"].sum()
			
			# Compute the document cosine similarity.
			results = doc_vectors.reset_index(name="bm25_sum")

			# Grab the top n results and store it to a heap.
			top_n = results.nlargest(max_results, "bm25_sum")
			top_n_list = []
			for _, row in top_n.iterrows():
				top_n_list.append((-row["bm25_sum"], row["doc"]))

			for doc_bm25_score, doc in top_n_list:
				# Insert the document name (includes file path), and 
				# BM25 (document sum) value to the heapq. The heapq 
				# sorts by the first value in the tuple so that is why 
				# the BM25 score is the first item in the tuple.
				if max_results != -1 and len(stack_heap) >= max_results:
					# Pushpop the highest (BM25) value tuple from the 
					# heap to make way for the next tuple.
					heapq.heappushpop(stack_heap, [doc_bm25_score, doc])
				else:
					# Push the highest (BM25) value tuple from the heap 
					# to make way for the next tuple.
					heapq.heappush(stack_heap, [doc_bm25_score, doc])

		return stack_heap


class VectorSearch:
	def __init__(self, index_dir: str, max_depth: int = 1, device: str = "cpu"):
		# Detect config.json file.
		assert os.path.exists("config.json"),\
			"Expected config.json file to exist. File is required for using vector search engine."
		
		# Verify that the model is supported with config.json.
		with open("config.json", "r") as f:
			config = json.load(f)

		self.config = config
		model = config["vector-search_config"]["model"]
		valid_models = config["models"]
		valid_model_names = list(valid_models.keys())
		assert model in valid_model_names,\
			f"Expected embedding model to be from valid models list {', '.join(valid_model_names)}. Received {model}."
		
		# Verify the model passed in matches the set model in the 
		# config.
		set_model = config["vector-search_config"]["model"]
		assert model == set_model,\
			f"Argument 'model' expected to match 'model' from 'vector-search_config' in config.json. Received {model}."

		# Load model config data.
		self.model_name = model
		self.model_config = valid_models[model]

		# Load model and tokenizer.
		self.device = device
		self.tokenizer, self.model = load_model(config, device=device)

		# Assert that the index directory path string is not empty.
		assert index_dir != "",\
			"Argument 'index_dir' expected to be a valid directory path."
		assert os.path.exists(index_dir), \
			f"Expected path to vector DB {index_dir} to exist."
		assert len(os.listdir(index_dir)) != 0, \
			f"Path to vector DB is empty."
		self.index_dir = index_dir
		self.max_depth = max_depth

		# Initialize (if need be) and connect to the vector database.
		uri = config["vector-search_config"]["db_uri"]
		self.db = lancedb.connect(uri)

		# Load model dims to pass along to the schema init.
		self.dims = config["models"][self.model_name]["dims"]
		
		# Open the table with the given table name.
		table_name = f"investopedia_depth{self.max_depth}"
		current_table_names = self.db.table_names()
		assert table_name in current_table_names,\
			f"Table {table_name} was expected to exist in database."
		
		# Initialize the fresh table for the current query.
		self.table = self.db.open_table(table_name)
		self.table.create_index(metric="cosine", vector_column_name="vector")


	def search(self, query: str, max_results: int = 50, document_ids: List = [], docs_are_results: bool = False):
		'''
		Conducts a search on the wikipedia data with vector search.
		@param: query str, the raw text that is being queried from the
			wikipedia data.
		@param: max_results (int), the maximum number of search results
			to return.
		@param: document_ids (List), the list of all document (paths)
			that are to be queried from the vector database/indices.
			Can also the the results list from stage 1 search if called
			from ReRank object.
		@param: docs_are_results (bool), a flag as to whether to the
			document_ids list passed in is actually stage 1 search 
			results. Default is False.
		@return: returns a list of objects where each object contains
			an article's path, title, retrieved text, and the slices of
			that text that is being returned (for BM25 and TF-IDF, 
			those slices values are for the whole article).
		'''
		# NOTE:
		# The original idea to compute embeddings on the fly was NOT a
		# good idea. Even with a few thousand documents, the embedding 
		# process ends up taking several hours to complete, 
		# illustrating how generating embeddings at runtime/inference 
		# does not scale. Since we're not dealing with a dataset on 
		# the scale of Wikipedia, precomputing vectors in the 
		# preprocess.py script is the best option. The only thing that 
		# will be embedded by the model will be the input query.
		# Additionally, a hard limit on the max_results does help too.

		# If the hard limit for the number of document ids or 
		# max_results is reached, print an error message and return an
		# empty results list.
		MAX_LIMIT = 10_000
		if len(document_ids) > MAX_LIMIT or max_results > MAX_LIMIT:
			print(f"Number of document_ids or max_results is too high. Hard limit of {MAX_LIMIT} for either.")
			return []
		
		# If the documents passed in are stage 1 search results, copy 
		# the results to their own variable and reset the document ids
		# list to be the document ids in those results.
		if docs_are_results:
			results = copy.deepcopy(document_ids)
			document_ids = [result[1] for result in results]

		# NOTE:
		# Assumes query text will exist within model tokenizer's max 
		# length. There might be complications for longer queries.
		print("Running Vector search...")

		# Embed the query text.
		query_embedding = self.embed_text(query)

		# Search the table.
		if len(document_ids) != 0:
			# Build the filter string based on the document ids.
			docs_for_query = ", ".join(
				f"'{document}'" for document in document_ids
			)

			results = self.table.search(query_embedding)\
				.where(f"file IN ({docs_for_query})")\
				.limit(max_results)\
				.to_list()
		else:
			results = self.table.search(query_embedding)\
				.limit(max_results)\
				.to_list()
	
		# Convert results to a list.
		assert results is not None
		assert isinstance(results, list)

		# Format search results.
		results = [
			tuple([
				result["_distance"], 
				result["file"], 
				load_article_text(result["file"]),
				[
					result["start_end_indices"][0],
					result["start_end_indices"][1]
				]
			])
			for result in results
		]

		# Return the search results.
		return results


	def embed_text(self, text: str):
		# Disable gradients.
		with torch.no_grad():
			# Pass original text chunk to tokenizer. Ensure the data is
			# passed to the appropriate (hardware) device.
			output = self.model(
				**self.tokenizer(
					text,
					add_special_tokens=False,
					padding="max_length",
					return_tensors="pt"
				).to(self.device)
			)

			# Compute the embedding by taking the mean of the last 
			# hidden state tensor across the seq_len axis.
			embedding = output[0].mean(dim=1)

			# Apply the following transformations to allow the
			# embedding to be compatible with being stored in the
			# vector DB (lancedb):
			#	1) Send the embedding to CPU (if it's not already
			#		there)
			#	2) Convert the embedding to numpy and flatten the
			# 		embedding to a 1D array
			embedding = embedding.to("cpu")
			embedding = embedding.numpy()[0]
		
		# Return the embedding.
		return embedding


class ReRankSearch:
	def __init__(self, bow_path: str, index_path: str,
			  	corpus_size: int = -1, avg_doc_len: float = -1.0,
				srt: float = -1.0, use_json: bool = False, 
				k1: float = 1.0, b: float = 0.0, device: str = "cpu",
				use_tf_idf: bool = False, max_depth: int = 1):
		# Set class variables.
		self.bow_dir = bow_path
		self.index_dir = index_path
		self.corpus_size = corpus_size
		self.avg_corpus_len = avg_doc_len
		self.srt = srt
		self.use_json = use_json
		self.k1 = k1
		self.b = b
		self.use_tfidf = use_tf_idf
		self.device = device
		self.max_depth = max_depth

		# Initialize search objects.
		self.tf_idf, self.bm25 = None, None
		if use_tf_idf:
			self.tf_idf = TF_IDF(
				self.bow_dir, self.corpus_size, self.srt, 
				use_json=self.use_json
			)
		else:
			self.bm25 = BM25(
				self.bow_dir, k1=self.k1, b=self.b, 
				corpus_size=self.corpus_size, 
				avg_doc_len=self.avg_corpus_len,
				srt=self.srt, use_json=self.use_json
			)
		self.vector_search = VectorSearch(
			self.index_dir, self.max_depth, self.device
		)

		# Organize search into stages.
		self.stage1 = self.tf_idf if self.use_tfidf else self.bm25
		self.stage2 = self.vector_search


	def search(self, query: str, max_results: int = 50):
		# Pass the search query to the first stage.
		stage_1_results = self.stage1.search(
			query, max_results=max_results
		)

		# Return the first stage search results if the results are empty.
		if len(stage_1_results) == 0:
			return stage_1_results

		# document_ids = [
		# 	# result["document_path"] for result in stage_1_results
		# 	result[1] for result in stage_1_results
		# ]

		# From the first stage, isolate the document paths to target in
		# the vector search.
		stage_2_results = self.stage2.search(
			query, max_results=max_results, document_ids=stage_1_results,
			docs_are_results=True
		)

		# Return the search results from the second stage.
		return stage_2_results