# preprocess.py
# Further preprocess the investopedia data. This will be important for 
# classical search algorithms like TF-IDF and BM25 as well as vector
# search.
# Python 3.11
# Windows/MacOS/Linux

import argparse
from argparse import Namespace
import copy
import gc
import json
import math
import multiprocessing as mp
import os
import pyarrow as pa
import shutil
import string
from typing import List, Dict, Tuple

from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import lancedb
import msgpack
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from num2words import num2words
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def process_page(page: Tag | NavigableString) -> str:
	'''
	Format and merge the texts from the <title> and <text> tags in the
		current <page> tag.
	@param: page (Tag | NavigatableString), the <page> element that is
		going to be parsed.
	@return: returns the text from the <title> and <text> tags merged
		together.
	'''

	# Assert correct typing for the page.
	assert isinstance(page, Tag) or isinstance(page, NavigableString) or page is None,\
		"Expected page to be a Tag or NavigatableString."

	# Return empty string if page is none.
	if page is None:
		return ""

	# Locate the title and text tags (expect to have 1 of each per 
	# article/page).
	text_tag = page.find("article")
	alt_text_tag = page.find(
		"header", 
		id="mntl-external-basic-sublayout_1-0"
	)

	# NOTE:
	# Anything not picked up by this process is considered "not a 
	# proper article".

	# Combine the title and text tag texts together.
	article_text = text_tag.get_text() if text_tag is not None else alt_text_tag.get_text()
	
	# Return the text.
	return article_text


def lowercase(text: str) -> str:
	'''
	Lowercase all characters in the text.
	@param: text (str), the text that is going to be lowercased.
	@return: returns the text with all characters in lowercase.
	'''
	return text.lower()


def handle_special_numbers(text: str) -> str:
	'''
	Replace all special numbers (circle digits, circle numbers, and
		parenthesis numbers) with more standardized representations 
		depending on the character.
	@param: text (str), the text that is going to have its text 
		removed/modified.
	@return: returns the text with all special numbers replaced with 
		regular numbers.
	'''
	# Mapping of circled digits to regular digits
	circled_digits = {
		'①': '(1)', '②': '(2)', '③': '(3)', '④': '(4)', '⑤': '(5)',
		'⑥': '(6)', '⑦': '(7)', '⑧': '(8)', '⑨': '(9)', '⑩': '(10)',
		'⑪': '(11)', '⑫': '(12)', '⑬': '(13)', '⑭': '(14)', '⑮': '(15)',
		'⑯': '(16)', '⑰': '(17)', '⑱': '(18)', '⑲': '(19)', '⑳': '(20)',
		'㉑': '(21)', '㉒': '(22)', '㉓': '(23)', '㉔': '(24)', '㉕': '(25)',
		'㉖': '(26)', '㉗': '(27)', '㉘': '(28)', '㉙': '(29)', '㉚': '(30)',
		'㉛': '(31)', '㉜': '(32)', '㉝': '(33)', '㉞': '(34)', '㉟': '(35)',
		'㊱': '(36)', '㊲': '(37)', '㊳': '(38)', '㊴': '(39)', '㊵': '(40)',
		'㊶': '(41)', '㊷': '(42)', '㊸': '(43)', '㊹': '(44)', '㊺': '(45)',
		'㊻': '(46)', '㊼': '(47)', '㊽': '(48)', '㊾': '(49)', '㊿': '(50)',
		'⓪': '(0)',
		'⓵': '(1)', '⓶': '(2)', '⓷': '(3)', '⓸': '(4)', '⓹': '(5)',
		'⓺': '(6)', '⓻': '(7)', '⓼': '(8)', '⓽': '(9)', '⓾': '(10)',
		'➀': '(1)', '➁': '(2)', '➂': '(3)',  '➃': '(4)',  '➄': '(5)',
		'➅': '(6)', '➆': '(7)', '➇': '(8)', '➈': '(9)', '➉': '(10)',
		'⓪': '(0)',
		'❶': '(1)', '❷': '(2)', '❸': '(3)', '❹': '(4)', '❺': '(5)',
        '❻': '(6)', '❼': '(7)', '❽': '(8)', '❾': '(9)', '❿': '(10)',
		'⓫': '(11)', '⓬': '(12)', '⓭': '(13)', '⓮': '(14)', '⓯': '(15)',
		'⓰': '(16)', '⓱': '(17)', '⓲': '(18)', '⓳': '(19)', '⓴': '(20)',
		'⓿': '(0)',
		'➊': '(1)', '➋': '(2)', '➌': '(3)', '➍': '(4)', '➎': '(5)',
		'➏': '(6)', '➐': '(7)', '➑': '(8)', '➒': '(9)', '➓': '(10)'
	}
	
	# Mapping of parenthesized digits to regular digits
	parenthesized_digits = {
		'⑴': '(1)', '⑵': '(2)', '⑶': '(3)', '⑷': '(4)', '⑸': '(5)',
		'⑹': '(6)', '⑺': '(7)', '⑻': '(8)', '⑼': '(9)', '⑽': '(10)',
		'⑾': '(11)', '⑿': '(12)', '⒀': '(13)', '⒁': '(14)', '⒂': '(15)',
		'⒃': '(16)', '⒄': '(17)', '⒅': '(18)', '⒆': '(19)', '⒇': '(20)',
		'⓪': '(0)'
	}
	
	# Combine both dictionaries
	all_special_numbers = {**circled_digits, **parenthesized_digits}
	
	# Replace all special numbers with their regular counterparts
	for special, regular in all_special_numbers.items():
		text = text.replace(special, regular)
	
	return text


def replace_superscripts(text: str) -> str:
	'''
	Replace all superscripts depending on the character.
	@param: text (str), the text that is going to have its text 
		removed/modified.
	@return: returns the text with all superscript characters 
		replaced with regular numbers.
	'''
	superscript_map = {
		'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', 
		'⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
	}
	
	# result = ""
	# for char in text:
	# 	if char in superscript_map:
	# 		# result += '^' + superscript_map[char]
	# 		result += superscript_map[char]
	# 	else:
	# 		result += char
	# return result
	result = []
	i = 0
	while i < len(text):
		if text[i] in superscript_map:
			# Start of a superscript sequence.
			sequence = []
			while i < len(text) and text[i] in superscript_map:
				sequence.append(superscript_map[text[i]])
				i += 1

			# Join the sequence and prepend "^".
			result.append('^' + ''.join(sequence) + " ")
		else:
			result.append(text[i])
			i += 1

	return ''.join(result)


def replace_subscripts(text: str) -> str:
	'''
	Replace all subscripts depending on the character.
	@param: text (str), the text that is going to have its text 
		removed/modified.
	@return: returns the text with all subscript characters replaced
		with regular numbers.
	'''
	subscript_map = {
		'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
		'₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
	}
	
	# result = ""
	# for char in text:
	# 	if char in superscript_map:
	# 		# result += '^' + superscript_map[char]
	# 		result += superscript_map[char]
	# 	else:
	# 		result += char
	# return result
	result = []
	i = 0
	while i < len(text):
		if text[i] in subscript_map:
			# Start of a superscript sequence.
			sequence = []
			while i < len(text) and text[i] in subscript_map:
				sequence.append(subscript_map[text[i]])
				i += 1

			# Join the sequence and prepend "^".
			result.append('^' + ''.join(sequence) + " ")
		else:
			result.append(text[i])
			i += 1

	return ''.join(result)


def remove_punctuation(text: str) -> str:
	'''
	Replace all punctuation with whitespace (" ") or empty space ("")
		depending on the character.
	@param: text (str), the text that is going to have its punctuation
		removed/modified.
	@return: returns the text with all punctuation a characters 
		replaced with whitespace or emptyspace.
	'''
	empty_characters = ",'"
	for char in string.punctuation:
		if char in empty_characters:
			text = text.replace(char, "")
		else:
			text = text.replace(char, " ")

	return text


def remove_stopwords(text: str) -> str:
	'''
	Remove all stopwords from the text.
	@param: text (str), the text that is going to have its stop words
		removed.
	@return: returns the text without any stop words.
	'''
	stop_words = set(stopwords.words("english"))
	words = word_tokenize(text)
	# new_text = ""
	# for word in words:
	# 	if word not in stop_words and len(word) > 1:
	# 		new_text = new_text + " " + word
	# text = new_text
	text = " ".join(
		[
			word for word in words 
			if word not in stop_words and len(word) > 1
		]
	)

	return text


def convert_numbers(text: str) -> str:
	'''
	Convert all numbers in the text from their numerical representation
		to their written expanded representation (1 -> one).
	@param: text (str), the text that is going to have its numbers 
		converted.
	@return: returns the text with all of its numbers expanded to their
		written expanded representations.
	'''
	words = word_tokenize(text)
	# new_text = ""
	# for w in words:
	# 	try:
	# 		w = num2words(int(w))
	# 	except:
	# 		a = 0
	# 	new_text = new_text + " " + w
	# new_text = np.char.replace(new_text, "-", " ")
	# text = new_text
	# text = " ".join(
	# 	[
	# 		num2words(int(word)) for word in words
	# 		if word.isdigit()
	# 	]
	# )

	# Define maximum number string length (10 ^ MAX_LEN).
	MAX_LEN = 307
	SIZE = MAX_LEN - 1

	# NOTE:
	# The Unicode range for New Tai Lue is 0x1980 to 0x19DF with the
	# range for digits being 19D0 to 19DF (source: 
	# https://en.wikipedia.org/wiki/New_Tai_Lue_(Unicode_block)).

	# Define the Unicode range for New Tai Lue digits.
	TAI_LUE_DIGIT_START = 0x19D0
	TAI_LUE_DIGIT_END = 0x19D9

	text = ""
	for word in words:
		# Check if the word is text within the unicode for new tai lue
		# digits.
		is_new_tai_lue_digit = any(
			TAI_LUE_DIGIT_START <= ord(char) <= TAI_LUE_DIGIT_END 
			for char in word
		)
		if word.isdigit() and is_new_tai_lue_digit:
			# Skip using num2words module if the word evaluates to
			# unicode in New Tai Lue digits.
			continue

		if word.isdigit():
			if len(word) < MAX_LEN:
				word = num2words(int(word))
			else:
				# Handles the edge case where the numerical text is 
				# greater than 1 x 10^307. Break apart the number into
				# chunks (of size/length 306 digits - NOT 307) and 
				# process each chunk before merging them together.
				chunked_number = [
					num2words(word[i:i + SIZE]) 
					for i in range(0, len(word), SIZE)
				]
				word = ' '.join(chunked_number)

		text = text + " " + word

	return text


def lemmatize(text: str) -> str:
	'''
	Lemmatize the words in the text.
	@param: text (str), the text that is going to be lemmatized.
	@return: returns the lemmatized text.
	'''
	lemmatizer = WordNetLemmatizer()
	words = word_tokenize(text)
	text = " ".join(
		[lemmatizer.lemmatize(word) for word in words]
	)

	return text


def stem(text) -> str:
	'''
	Stem the words in the text.
	@param: text (str), the text that is going to be stemmed.
	@return: returns the stemmed text.
	'''
	stemmer = PorterStemmer()
	words = word_tokenize(text)
	text = " ".join(
		[stemmer.stem(word) for word in words]
	)
	
	return text


def bow_preprocessing(text: str, return_word_freq: bool=False) -> Tuple[List[str], Dict[str, int]] | Tuple[List[str]]:
	'''
	Preprocess the text to yield a bag of words used in the text. Also
		compute the word frequencies for each word in the bag of words
		if specified in the arguments.
	@param: text (str), the raw text that is to be processed into a bag
		of words.
	@param: return_word_freq (bool), whether to return the frequency of
		each word in the input text.
	@return: returns a tuple (bag_of_words: List[str]) or 
		(bag_of_words: List[str], freq: dict[str: int]) depending on the 
		return_word_freq argument.
	'''
	# Perform the following text preprocessing in the following order:
	# 1) lowercase
	# 2) handle special (circle) numbers 
	# 3) remove punctuation
	# 4) remove stop words
	# 5) remove superscripts/subscripts
	# 6) convert numbers
	# 7) lemmatize
	# 8) stem
	# 9) remove punctuation
	# 10) convert numbers
	# 11) stem
	# 12) remove punctuation
	# 13) remove stopwords
	# Note how some of these steps are repeated. This is because 
	# previous steps may have introduced conditions that were 
	# previously handled. However, this order of operations is both
	# optimal and firm.
	text = lowercase(text)
	text = handle_special_numbers(text)
	text = remove_punctuation(text)
	text = remove_stopwords(text)
	text = replace_superscripts(text)
	text = replace_subscripts(text)
	text = convert_numbers(text)
	text = lemmatize(text)
	text = stem(text)
	text = remove_punctuation(text)
	text = convert_numbers(text)
	text = stem(text)
	text = remove_punctuation(text)
	text = remove_stopwords(text)

	# NOTE:
	# Number conversion, lemmatizaton, and stemming do not seem to be
	# hard/firm requirements to perform the bag of words preprocessing.
	# They do offer some improvement for the search component
	# downstream so they are left in.
	# One aspect that does have me wary of performing this is that
	# these steps rely on third party libraries to do their respective
	# transformations. Since I want to convert this project to JS, I am
	# trying to minimize the number of third party packages since I
	# have to find a JS counterpart that works as close as possible.

	# Isolate the set of unique words in the remaining processed text.
	bag_of_words = list(set(word_tokenize(text)))

	# Return just the bag of words if return_word_freq is False.
	if not return_word_freq:
		return tuple([bag_of_words])

 	# Record each word's frequency in the processed text.
	word_freqs = dict()
	words = word_tokenize(text)
	for word in bag_of_words:
		word_freqs[word] = words.count(word)

	# Return the bag of words and the word frequencies.
	return tuple([bag_of_words, word_freqs])


def load_model(config: Dict, device="cpu") -> Tuple[AutoTokenizer, AutoModel]:
	'''
	Load the tokenizer and model. Download them if they're not found 
		locally.
	@param: config (Dict), the configuration JSON. This will specify
		the model and its path attributes.
	@param: device (str), tells where to map the model. Default is 
		"cpu".
	@return: returns the tokenizer and model for embedding the text.
	'''
	# Check for the local copy of the model. If the model doesn't have
	# a local copy (the path doesn't exist), download it.
	model_name = config["vector-search_config"]["model"]
	model_config = config["models"][model_name]
	model_path = model_config["storage_dir"]
	
	# Check for path and that path is a directory. Make it if either is
	# not true.
	if not os.path.exists(model_path) or not os.path.isdir(model_path):
		os.makedirs(model_path, exist_ok=True)

	# Check for path the be populated with files (weak check). Download
	# the tokenizer and model and clean up files once done.
	if len(os.listdir(model_path)) == 0:
		print(f"Model {model_name} needs to be downloaded.")

		# Check for internet connection (also checks to see that
		# huggingface is online as well). Exit if fails.
		response = requests.get("https://huggingface.co/")
		if response.status_code != 200:
			print(f"Request to huggingface.co returned unexpected status code: {response.status_code}")
			print(f"Unable to download {model_name} model.")
			exit(1)

		# Create cache path folders.
		cache_path = model_config["cache_dir"]
		os.makedirs(cache_path, exist_ok=True)
		os.makedirs(model_path, exist_ok=True)

		# Load tokenizer and model.
		model_id = model_config["model_id"]
		tokenizer = AutoTokenizer.from_pretrained(
			model_id, cache_dir=cache_path, device_map=device
		)
		model = AutoModel.from_pretrained(
			model_id, cache_dir=cache_path, device_map=device
		)

		# Save the tokenizer and model to the save path.
		tokenizer.save_pretrained(model_path)
		model.save_pretrained(model_path)

		# Delete the cache.
		shutil.rmtree(cache_path)
	
	# Load the tokenizer and model.
	tokenizer = AutoTokenizer.from_pretrained(
		model_path, device_map=device
	)
	model = AutoModel.from_pretrained(
		model_path, device_map=device
	)

	# Return the tokenizer and model.
	return tokenizer, model


def vector_preprocessing(article_text: str, config: Dict, tokenizer: AutoTokenizer) -> List[Dict]:
	'''
	Preprocess the text to yield a list of chunks of the tokenized 
		text. Each chunk is the longest possible set of text that can 
		be passed to the embedding model tokenizer.
	@param: text (str), the raw text that is to be processed for
		storing to vector database.
	@param: config (dict), the configuration parameters. These 
		parameters detail important parts of the vector preprocessing
		such as context length.
	@param: tokenizer (AutoTokenizer), the tokenizer for the embedding
		model.
	@return: returns a List[Dict] of the text metadata. This metadata 
		includes the split text's token sequence, index (with respect
		to the input text), and length of the text split for each split
		in the text.
	'''
	# Pull the model's context length and overlap token count from the
	# configuration file.
	model_name = config["vector-search_config"]["model"]
	model_config = config["models"][model_name]
	context_length = model_config["max_tokens"]
	overlap = config["preprocessing"]["token_overlap"]

	# Make sure that the overlap does not exceed the model context
	# length.
	assert overlap < context_length, f"Number of overlapping tokens ({overlap}) must NOT exceed the model context length ({context_length})"

	split_text = article_text.split("\n\n")
	while "\n" in split_text or "" in split_text:
		if "\n" in split_text:
			split_text.remove("\n")
		
		if "" in split_text:
			split_text.remove("")

	all_tokens = list()
	for text_chunk in split_text:
		# Tokenize.
		tokens = tokenizer(
			text_chunk, 
			return_attention_mask=True,
			return_offsets_mapping=True,
			# return_tensors="pt",
			# padding="max_length",
		)
		print(tokens)
		# all_tokens.append(tokens)

		input_ids = tokens["input_ids"]
		token_type_ids = tokens["token_type_ids"]
		attention_mask = tokens["attention_mask"]
		offset_mapping = tokens["offset_mapping"]
		pad_token_id = 0

		start_idx = 0

		while start_idx < len(input_ids):
			end_idx = min(start_idx + context_length, len(input_ids))
			
			# Extract data from the current chunk
			chunk_input_ids = input_ids[start_idx:end_idx]
			chunk_type_ids = token_type_ids[start_idx:end_idx]
			chunk_attention_mask = attention_mask[start_idx:end_idx]
			chunk_offset_mappings = offset_mapping[start_idx:end_idx]
			
			# Determine the start and end character indices in the 
			# original text.
			chunk_start = chunk_offset_mappings[0][0]
			chunk_end = chunk_offset_mappings[-1][-1]

			# Pad the last chunk if necessary
			if len(chunk_input_ids) < context_length:
				chunk_input_ids += [pad_token_id] * (context_length - len(chunk_input_ids))
			
			all_tokens.append(
				{	
					"input_ids": chunk_input_ids, 
					"attention_mask": chunk_attention_mask,
					"chunk_type_ids": chunk_type_ids,
					"text_range": (chunk_start, chunk_end)
				}
			)

			# MOve the start index forward, considering the overlap.
			start_idx = end_idx - overlap  # Move forward, keeping the overlap

		# all_tokens.append(tokens)
		
	return all_tokens


	# metadata = []

	# # Add to the metadata list by passing the text to the high level
	# # recursive splitter function.
	# metadata += high_level_split(
	# 	article_text, 0, tokenizer, context_length, splitters
	# )
	
	# # Return the text metadata.
	# return metadata


def merge_mappings(results: List[List]) -> Tuple[Dict, Dict]:
	'''
	Merge the results of processing each article in the file from the 
		multiprocessing pool.
	@param: results (list[list]), the list containing the outputs of
		the processing function for each processor.
	@return: returns a tuple of the same processing outputs now 
		aggregated together.
	'''
	# Initialize aggregate variables.
	aggr_word_to_doc = dict()
	aggr_doc_to_word = dict()

	# Results mappings shape (num_processors, tuple_len). Iterate
	# through each result and update the aggregate variables.
	for result in results:
		# Unpack the result tuple.
		doc_to_word, word_to_doc, _ = result

		# Iteratively update the word to document dictionary.
		for key, value in word_to_doc.items():
			assert isinstance(value, int)
			if key not in aggr_word_to_doc:
				aggr_word_to_doc[key] = value
			else:
				aggr_word_to_doc[key] += value

		# Update the document to word dictionary. Just call a
		# dictionary's update() function here since every key in the
		# entirety of the results is unique.
		aggr_doc_to_word.update(doc_to_word)

	# Return the aggregated data.
	return aggr_doc_to_word, aggr_word_to_doc


def multiprocess_articles(
	args: Namespace, 
	device: str, 
	data_files: List[str], 
	num_proc: int = 1
) -> Tuple[Dict, Dict]:
	'''
	Preprocess the text (in multiple processors).
	@param: args (Namespace), the arguments passed in from the 
		terminal.
	@param: device (str), the name of the CPU or GPU device that the
		embedding model will use.
	@param: data_files (List[str]), the list of filepaths being 
		processed.
	@param: num_proc (int), the number of processes to use. Default is 
		1.
	@return: returns the set of dictionaries containing the necessary 
		data and metadata to index the articles.
	'''
	# Reset args.vector to False because lancedb doesn't work well with
	# multiprocessing/inserting vectors into tables under subprocesses.
	args.vector = False

	# Break down the list of pages into chunks.
	chunk_size = math.ceil(len(data_files) / num_proc)
	chunks = [
		data_files[i:i + chunk_size] 
		for i in range(0, len(data_files), chunk_size)
	]

	# Define the arguments list.
	arg_list = [(args, device, chunk) for chunk in chunks]

	# Distribute the arguments among the pool of processes.
	with mp.Pool(processes=num_proc) as pool:
		# Aggregate the results of processes.
		results = pool.starmap(process_articles, arg_list)

		# Pass the aggregate results tuple to be merged.
		doc_to_word, word_to_doc = merge_mappings(
			results
		)

	# Return the different mappings.
	return doc_to_word, word_to_doc


def process_articles(
	args: Namespace, 
	device: str,
	data_files: List[str]
) -> Tuple[Dict, Dict]:
	'''
	Preprocess the text (in a single thread/process).
	@param: args (Namespace), the arguments passed in from the 
		terminal.
	@param: device (str), the name of the CPU or GPU device that the
		embedding model will use.
	@param: data_files (List[str]), the list of filepaths being 
		processed.
	@return: returns the set of dictionaries and list containing the 
		necessary data and metadata to index the articles.
	'''
	# Initialize local mappings.
	doc_to_word = dict()
	word_to_doc = dict()
	vector_metadata = list()

	# Load the configurations from the config JSON.
	with open("config.json", "r") as f:
		config = json.load(f)

	# Load the model tokenizer and model (if applicable). Do this here
	# instead of within the for loop for (runtime) efficiency.
	tokenizer, model = None, None
	db, table = None, None

	for file in tqdm(data_files):
		# Load the file and pass it to beautifulsoup.
		with open(file, "r") as f:
			page = BeautifulSoup(f.read(), "lxml")

		# Isolate the article/page's raw text. Create copies for each
		# preprocessing task.
		try:
			article_text = process_page(page)
		except:
			print(f"Unable to parse invalid article: {file}")
			continue

		###############################################################
		# BAG OF WORDS
		###############################################################
		if args.bow:
			# Create a copy of the raw text.
			article_text_bow = copy.deepcopy(article_text)

			# Create a bag of words for each article (xml) file.
			xml_bow, xml_word_freq = bow_preprocessing(
				article_text_bow, return_word_freq=True
			)

			# Update word to document map.
			for word in xml_bow:
				if word in list(word_to_doc.keys()):
					word_to_doc[word] += 1
				else:
					word_to_doc[word] = 1

			# Update the document to words map.
			doc_to_word[file] = xml_word_freq

		###############################################################
		# VECTOR EMBEDDINGS
		###############################################################
		# if args.vector:
		# 	# Load the tokenizer and model.
		# 	tokenizer, model = load_model(config, device)

		# 	# Connect to the vector database
		# 	uri = config["vector-search_config"]["db_uri"]
		# 	db = lancedb.connect(uri)

		# 	# Assert the table for the file exists (should have been
		# 	# initialized in the for loop in main()).
		# 	table_name = os.path.basename(file).rstrip(".html")
		# 	current_tables = db.table_names()
		# 	assert table_name in current_tables,\
		# 		f"Expected table {table_name} to be in list of current tables before vector embedding preprocessing.\nCurrent Tables: {', '.join(current_tables)}"

		# 	# Get the table for the file.
		# 	table = db.open_table(table_name)

		# 	# Assertion to make sure tokenizer and model and vector
		# 	# database and current table are initialized.
		# 	assert None not in [tokenizer, model], "Model tokenizer and model is expected to be initialized for vector embeddings preprocessing."
		# 	assert None not in [db, table], "Vector database and current table are expected to be initialized for vector embeddings preprocessing."

		# 	# Create a copy of the raw text.
		# 	article_text_v_db = copy.deepcopy(article_text)

		# 	# Pass the article to break the text into manageable chunks
		# 	# for the embedding model. This will yield the (padded) 
		# 	# token sequences for each chunk as well as the chunk 
		# 	# metadata (such as the respective index in the original
		# 	# text for each chunk and the length of the chunk).
		# 	# chunk_metadata = vector_preprocessing(
		# 	# 	article_text_v_db, config, tokenizer
		# 	# )
		# 	chunk_metadata = None

		# 	# Disable gradients.
		# 	with torch.no_grad():
		# 		# Embed each chunk and update the metadata.
		# 		for idx, chunk in enumerate(chunk_metadata):
		# 			# Update/add the metadata for the source filename
		# 			# and article SHA1.
		# 			chunk.update({"file": file})

		# 			# Get original text chunk from text.
		# 			text_idx = chunk["text_idx"]
		# 			text_len = chunk["text_len"]
		# 			text_chunk = article_text_v_db[text_idx: text_idx + text_len]

		# 			# Pass original text chunk to tokenizer. Ensure the
		# 			# data is passed to the appropriate (hardware)
		# 			# device.
		# 			output = model(
		# 				**tokenizer(
		# 					text_chunk,
		# 					add_special_tokens=False,
		# 					padding="max_length",
		# 					return_tensors="pt"
		# 				).to(device)
		# 			)

		# 			# Compute the embedding by taking the mean of the
		# 			# last hidden state tensor across the seq_len axis.
		# 			embedding = output[0].mean(dim=1)

		# 			# Apply the following transformations to allow the
		# 			# embedding to be compatible with being stored in 
		# 			# the vector DB (lancedb):
		# 			#	1) Send the embedding to CPU (if it's not 
		# 			#		already there)
		# 			#	2) Convert the embedding to numpy and flatten 
		# 			# 		the embedding to a 1D array
		# 			embedding = embedding.to("cpu")
		# 			embedding = embedding.numpy()[0]

		# 			# NOTE:
		# 			# Originally I had embeddings stored into the 
		# 			# metadata dictionary under the "embedding", key
		# 			# but lancedb requires the embedding data be under 
		# 			# the "vector" name.

		# 			# Update the chunk dictionary with the embedding
		# 			# and set the value of that chunk in the metadata
		# 			# list to the (updated) chunk.
		# 			# chunk.update({"embedding": embedding})
		# 			chunk.update({"vector": embedding})
		# 			chunk_metadata[idx] = chunk
				
		# 	# Add the chunk metadata to the vector metadata.
		# 	# vector_metadata += chunk_metadata

		# 	# Add chunk metadata to the vector database. Should be on
		# 	# "append" mode by default.
		# 	table.add(chunk_metadata, mode="append")
	
	# Return the mappings and metadata.
	return doc_to_word, word_to_doc, vector_metadata


def main() -> None:
	'''
	Main method. Process the individual wikipedia articles from their
		xml files to create document to word and word to document 
		mappings for faster bag of words processing during classical
		search (TF-IDF/BM25) and vector databases for vector search.
	@param: takes no arguments.
	@return: returns nothing.
	'''
	###################################################################
	# PROGRAM ARGUMENTS
	###################################################################
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--restart",
		action="store_true",
		help="Specify whether to restart the preprocessing from scratch. Default is false/not specified."
	)
	parser.add_argument(
		"--bow",
		action="store_true",
		help="Specify whether to run the bag-of-words preprocessing. Default is false/not specified."
	)
	parser.add_argument(
		"--vector",
		action="store_true",
		help="Specify whether to run the vector database preprocessing. Default is false/not specified."
	)
	parser.add_argument(
		"--max_depth",
		type=int,
		default=1,
		help="How deep should the graph traversal go across links. Default is 1/not specified."
	)
	parser.add_argument(
		'--num_proc', 
		type=int, 
		default=1, 
		help="Number of processor cores to use for multiprocessing. Default is 1."
	)
	parser.add_argument(
		'--gpu2cpu_limit', 
		type=int, 
		default=4, 
		help="Maximum number of processor cores allowed before GPU is disabled. Default is 4."
	)
	parser.add_argument(
		'--override_gpu2cpu_limit', 
		action='store_true', 
		help="Whether to override the gpu2cpu_proc value. Default is false/not specified."
	)
	args = parser.parse_args()

	###################################################################
	# VERIFY DATA FILES
	###################################################################
	# Check for InvestopediaDownload submodule and additional necessary
	# folders to be initialized.
	submodule_dir = "./InvestopediaDownload"
	submodule_data_dir = os.path.join(submodule_dir, "data")
	submodule_graph_dir = os.path.join(submodule_dir, "graph")
	if not os.path.exists(submodule_dir):
		print(f"InvestopediaDownload submodule not initialized.")
		print(f"Please initialized submodule with 'git submodule update --init --recursive'")
		exit(1)
	elif not os.path.exists(submodule_data_dir) or not os.path.exists(submodule_graph_dir):
		print(f"InvestopediaDownload submodule has not extracted any articles from the downloader.")
		print(f"Follow the README.md in the InvestopediaDownload submodule for instructions on how to download and extract articles from wikipedia.")
		exit(1)
		
	# Check for necessary graph files to be initialized.
	assert args.max_depth > 0, \
		f"Invalid --max_depth value was passed in (must be > 0, recieved {args.max_depth})"
	if args.max_depth > 1:
		graph_file = os.path.join(
			submodule_graph_dir,
			f"term_article_graph_depth{args.max_depth}.json"
		)
		expanded_article_map_file = os.path.join(
			submodule_graph_dir,
			f"expanded_article_map_depth{args.max_depth}.json"
		)
	else:
		graph_file = None
		expanded_article_map_file = os.path.join(
			submodule_graph_dir,
			f"article_map.json"
		)
	if graph_file is not None and not os.path.exists(graph_file):
		print(f"InvestopediaDownload submodule has not extracted any articles from the downloader.")
		print(f"This is signified by the missing {graph_file} for graph of depth {args.max_depth}.")
		print(f"Follow the README.md in the InvestopediaDownload submodule for instructions on how to download and extract articles from wikipedia.")
		exit(1)
	elif not os.path.exists(expanded_article_map_file):
		print(f"InvestopediaDownload submodule has not extracted any articles from the downloader.")
		print(f"This is signified by the missing {expanded_article_map_file} for article map of depth {args.max_depth}.")
		print(f"Follow the README.md in the InvestopediaDownload submodule for instructions on how to download and extract articles from wikipedia.")
		exit(1)
	
	# Load (expanded) article map.
	with open(expanded_article_map_file, "r") as f:
		expanded_article_map = json.load(f)
	
	# NOTE:
	# I tried to make this cleaner but python would throw an error on
	# on the os.listdir() line for the submodule data directory if that
	# directory did not exist. Therefore, it made it impossible to
	# define data_files before checking for the existance of the 
	# required submodule data directory.
	data_files = sorted(
		[
			os.path.join(
				submodule_data_dir, 
				expanded_article_map[article]["path"]
			)
			for article in expanded_article_map
		]
	)
	data_files = [
		data_file.replace("./data/", "") 
		for data_file in data_files
	] # Clean up path (extra "./data/" in the path string)
	data_files = [
		data_file for data_file in data_files
		if os.path.exists(data_file)
	] # Only include paths that exist
	if len(data_files) == 0:
		print(f"InvestopediaDownload submodule has not extracted any articles from the downloader.")
		print(f"Follow the README.md in the InvestopediaDownload submodule for instructions on how to download and extract articles from wikipedia.")
		exit(1)

	###################################################################
	# NLTK SETUP
	###################################################################
	# Download packages from nltk.
	nltk.download("stopwords")

	###################################################################
	# EMBEDDING MODEL SETUP
	###################################################################
	# Load the configurations from the config JSON.
	with open("config.json", "r") as f:
		config = json.load(f)

	# Check for embedding model files and download them if necessary.
	load_model(config)

	###################################################################
	# VECTOR DB SETUP
	###################################################################
	# Initialize (if need be) and connect to the vector database.
	uri = config["vector-search_config"]["db_uri"]
	db = lancedb.connect(uri)

	# Load model dims to pass along to the schema init.
	model_name = config["vector-search_config"]["model"]
	dims = config["models"][model_name]["dims"]

	# Initialize schema (this will be passed to the database when 
	# creating a new, empty table in the vector database).
	schema = pa.schema([
		pa.field("file", pa.utf8()),
		pa.field("text_idx", pa.int32()),
		pa.field("text_len", pa.int32()),
		pa.field("vector", pa.list_(pa.float32(), dims))
	])

	###################################################################
	# METADATA PATHS
	###################################################################
	# Pull directory paths from the config file.
	preprocessing = config["preprocessing"]
	d2w_metadata_path = os.path.join(
		preprocessing["doc_to_words_path"],
		f"depth_{args.max_depth}"
	)
	w2d_metadata_path = os.path.join(
		preprocessing["word_to_docs_path"],
		f"depth_{args.max_depth}"
	)
	vector_metadata_path = os.path.join(
		preprocessing["vector_metadata_path"],
		f"depth_{args.max_depth}"
	)
	
	# Initialize the directories if they don't already exist.
	if not os.path.exists(d2w_metadata_path):
		os.makedirs(d2w_metadata_path, exist_ok=True)

	if not os.path.exists(w2d_metadata_path):
		os.makedirs(w2d_metadata_path, exist_ok=True)

	if not os.path.exists(vector_metadata_path):
		os.makedirs(vector_metadata_path, exist_ok=True)

	###################################################################
	# RESTART CHECK
	###################################################################
	doc2words_path_json = os.path.join(
		d2w_metadata_path, 
		"doc2words.json"
	)
	doc2words_path_msgpack = os.path.join(
		d2w_metadata_path, 
		"doc2words.msgpack"
	)
	word2docs_path_json = os.path.join(
		w2d_metadata_path, 
		"word2docs.json"
	)
	word2docs_path_msgpack = os.path.join(
		w2d_metadata_path, 
		"word2docs.msgpack"
	)

	if args.restart:
		# Clear the progress files if the restart flag has been thrown.
		open(doc2words_path_json, "w+").close()
		open(doc2words_path_msgpack, "w+").close()
		open(word2docs_path_json, "w+").close()
		open(word2docs_path_msgpack, "w+").close()

	###################################################################
	# FILE PREPROCESSING
	###################################################################
	# Initialize a dictionary to keep track of the word to documents
	# and documents to words mappings.
	word_to_doc = dict()
	doc_to_word = dict()

	# NOTE:
	# Be careful if you are on mac. Because Apple Silicon works off of
	# the unified memory model, there may be some performance hit for 
	# CPU bound tasks. The hope is that MPS will actually accelerate 
	# the embedding model's performance.

	# GPU setup.
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"

	# Unpack arguments for multiprocessing.
	num_proc = args.num_proc
	override_gpu2cpu = args.override_gpu2cpu_limit
	gpu2cpu_limit = args.gpu2cpu_limit

	if num_proc > 1:
		# Determine the number of CPU cores to use (this will be
		# passed down the the multiprocessing function)
		max_proc = min(mp.cpu_count(), num_proc)

		# Reset the device if the number of processes to be used is
		# greater than 4. This is because the device setting is
		# quite rudimentary with this system. I don't know
		# 1) How much VRAM each instance of a model would take up 
		#	vs the amount of VRAM available (4GB, 8GB, 12GB, ...).
		# 2) How transformers or pytorch would have to be 
		#	configured to balance the number of model instances on
		#	each process against multiple GPUs on device.
		# This is also still assuming that with multiprocessing
		# enabled, the user has a sufficient regular memory/RAM to
		# load everything there. For now, this just makes things
		# simpler.
		if max_proc > gpu2cpu_limit and not override_gpu2cpu:
			device = "cpu"

		doc_to_word, word_to_doc = multiprocess_articles(
			args, device, data_files, num_proc=max_proc
		)
	else:
		doc_to_word, word_to_doc, _ = process_articles(
			args, device, data_files
		)

	vector_metadata = list()

	# Load the configurations from the config JSON.
	with open("config.json", "r") as f:
		config = json.load(f)

	# Load the model tokenizer and model (if applicable). Do this here
	# instead of within the for loop for (runtime) efficiency.
	tokenizer, model = None, None
	db, table = None, None

	for file in tqdm(data_files):
		# Load the file and pass it to beautifulsoup.
		with open(file, "r") as f:
			page = BeautifulSoup(f.read(), "lxml")

		# Isolate the article/page's raw text. Create copies for each
		# preprocessing task.
		try:
			article_text = process_page(page)
		except:
			print(f"Unable to parse invalid article: {file}")
			continue

		###############################################################
		# VECTOR EMBEDDINGS
		###############################################################
		if args.vector:
			# Load the tokenizer and model.
			tokenizer, model = load_model(config, device)

			# Connect to the vector database
			uri = config["vector-search_config"]["db_uri"]
			db = lancedb.connect(uri)

			# Assert the table for the file exists (should have been
			# initialized in the for loop in main()).
			# table_name = os.path.basename(file).rstrip(".html")
			current_tables = db.table_names()
			table_name = f"investopedia_depth{args.max_depth}"
			if table_name not in current_tables:
				db.create_table(table_name, schema=schema)
			# assert table_name in current_tables,\
			# 	f"Expected table {table_name} to be in list of current tables before vector embedding preprocessing.\nCurrent Tables: {', '.join(current_tables)}"

			# Get the table for the file.
			table = db.open_table(table_name)

			# Assertion to make sure tokenizer and model and vector
			# database and current table are initialized.
			assert None not in [tokenizer, model], "Model tokenizer and model is expected to be initialized for vector embeddings preprocessing."
			assert None not in [db, table], "Vector database and current table are expected to be initialized for vector embeddings preprocessing."

			# Create a copy of the raw text.
			article_text_v_db = copy.deepcopy(article_text)

			# Pass the article to break the text into manageable chunks
			# for the embedding model. This will yield the (padded) 
			# token sequences for each chunk as well as the chunk 
			# metadata (such as the respective index in the original
			# text for each chunk and the length of the chunk).
			chunk_metadata = vector_preprocessing(
				article_text_v_db, config, tokenizer
			)
			# chunk_metadata = None

			# Disable gradients.
			with torch.no_grad():
				# Embed each chunk and update the metadata.
				for idx, chunk in enumerate(chunk_metadata):
					# Update/add the metadata for the source filename
					# and article SHA1.
					chunk.update({"file": file})

					# Get original text chunk from text.
					# text_idx = chunk["text_idx"]
					# text_len = chunk["text_len"]
					# text_chunk = article_text_v_db[text_idx: text_idx + text_len]

					# Pass original text chunk to tokenizer. Ensure the
					# data is passed to the appropriate (hardware)
					# device.
					output = model(
						# **tokenizer(
						# 	text_chunk,
						# 	add_special_tokens=False,
						# 	padding="max_length",
						# 	return_tensors="pt"
						# ).to(device)
						**chunk
					)

					# Compute the embedding by taking the mean of the
					# last hidden state tensor across the seq_len axis.
					embedding = output[0].mean(dim=1)

					# Apply the following transformations to allow the
					# embedding to be compatible with being stored in 
					# the vector DB (lancedb):
					#	1) Send the embedding to CPU (if it's not 
					#		already there)
					#	2) Convert the embedding to numpy and flatten 
					# 		the embedding to a 1D array
					embedding = embedding.to("cpu")
					embedding = embedding.numpy()[0]

					# NOTE:
					# Originally I had embeddings stored into the 
					# metadata dictionary under the "embedding", key
					# but lancedb requires the embedding data be under 
					# the "vector" name.

					# Update the chunk dictionary with the embedding
					# and set the value of that chunk in the metadata
					# list to the (updated) chunk.
					# chunk.update({"embedding": embedding})
					chunk.update({"vector": embedding})
					chunk_metadata[idx] = chunk
				
			# Add the chunk metadata to the vector metadata.
			# vector_metadata += chunk_metadata

			# Add chunk metadata to the vector database. Should be on
			# "append" mode by default.
			table.add(chunk_metadata, mode="append")

	###############################################################
	# SAVE MAPPINGS.
	###############################################################

	# Write metadata to the respective files.
	if len(list(doc_to_word.keys())) > 0:
		with open(doc2words_path_json, "w+") as d2w_f:
			json.dump(doc_to_word, d2w_f, indent=4)
		with open(doc2words_path_msgpack, "wb+") as d2w_f:
			packed = msgpack.packb(doc_to_word)
			d2w_f.write(packed)

	if len(list(word_to_doc.keys())) > 0:
		with open(word2docs_path_json, "w+") as w2d_f:
			json.dump(word_to_doc, w2d_f, indent=4)
		with open(word2docs_path_msgpack, "wb+") as w2d_f:
			packed = msgpack.packb(word_to_doc)
			w2d_f.write(packed)

	# Perform garbage collection.
	gc.collect()

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	# Required to initialize models on GPU for multiprocessing. Placed
	# here due to recommendation from official python documentation.
	mp.set_start_method("spawn", force=True)
	main()