# searchwiki.py
# Run a search on the downloaded wikipedia data. The wikipediate data
# should already be downloaded, extracted, and preprocessed by the
# WikipediaEnDownload submodule as well as preprocess.py in this repo.
# Python 3.9
# Windows/MacOS/Linux


import argparse
from argparse import Namespace
import json
import math
import os
import random
import shutil
import time

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from preprocess import process_page
from search import TF_IDF, BM25, VectorSearch, ReRankSearch
from search import print_results


def build_query(text: str, device: str = "cpu") -> str:
	'''
	Build a query with a given text snippet using a small LLM (such as 
		llama 3.2 1B).
	@param: text (str), the text that is going to be analyzed and 
		answered with the output query.
	@param: device (str), which hardware acceleration to use (if any)
		for the small LLM. Default is "cpu".
	@return: returns a query string that is answered by the input text.
	'''
	# Model config.
	model_id = "meta-llama/Llama-3.2-1B-Instruct"
	model_cache = "./llama/cache/" + model_id.replace("/", "_")
	model_path = "./llama/models/" + model_id.replace("/", "_")
	os.makedirs(model_cache, exist_ok=True)
	os.makedirs(model_path, exist_ok=True)

	if len(os.listdir(model_path)) == 0:
		# Load huggingface token (required for downloading llama 3.2 1b 
		# instruct model).
		env_file = ".env"
		if not os.path.exists(env_file):
			print(f"Unable to locate '.env' file. File is required to download the model from huggingface.")
			exit(1)

		with open(env_file, "r") as f:
			token = f.readline().replace("\n", "")

		# Initialize tokenizer and model. Download both to the cache
		# directory and save to the model directory.
		tokenizer = AutoTokenizer.from_pretrained(
			model_id, cache_dir=model_cache, use_auth_token=token
		)
		model = AutoModelForCausalLM.from_pretrained(
			model_id, cache_dir=model_cache, use_auth_token=token
		)
		tokenizer.save_pretrained(model_path)
		model.save_pretrained(model_path)

		# Clear the cache path.
		shutil.rmtree(model_cache)

	# Initialize pipeline.
	pipe = pipeline(
		"text-generation",
		model=AutoModelForCausalLM.from_pretrained(model_path),
		tokenizer=AutoTokenizer.from_pretrained(model_path),
		device=device
	)

	# Initialize instruction messages.
	messages = [
		{
			"role": "system", 
			"content": "You are a helpful chatbot that follows instructions exactly and concisely."
		},
		{
			"role": "user", 
			"content": f"""
				Given the following text, write a question that can be answered with the following input text. Output the response in the following format:
				
				- Question: [question]
				- Answer: [answer from the text]
				
				Input Text:
				{text[0] if isinstance(text, list) else text}
			"""
		},
	]

	# Pass instruction into the pipeline.
	outputs = pipe(messages, max_new_tokens=1024)

	# Isolate the response for the user. Clean up the output and return 
	# the query.
	query = outputs[0]["generated_text"][-1]
	query = query["content"].split("\n")[0].replace("- Question:", "")
	return query


def test(args: Namespace) -> None:
	'''
	Test each of the search processes on the wikipedia dataset.
	@param: takes no arguments.
	@return: returns nothing.
	'''
	# Input values to search engines.
	with open("config.json", "r") as f:
		config = json.load(f)

	bow_dir = "./metadata/bag_of_words"
	index_dir = "./vector_data/vector_db"

	assert args.max_depth > 0, \
		f"Invalid --max_depth value was passed in (must be > 0, recieved {args.max_depth})"
	
	preprocessing_paths = config["preprocessing"]
	corpus_staging = os.path.join(
		preprocessing_paths["staging_corpus_path"], 
		f"depth_{args.max_depth}"
	)
	corpus_path = os.path.join(corpus_staging, "corpus_stats.json")

	# Load corpus stats from the corpus path JSON if it exists. Use the
	# config JSON if the corpus JSON is not available.
	if os.path.exists(corpus_path):
		with open(corpus_path, "r") as f:
			corpus_stats = json.load(f)
			tfidf_corpus_size = corpus_stats["corpus_size"]
			bm25_corpus_size = corpus_stats["corpus_size"]
			bm25_avg_doc_len = corpus_stats["avg_doc_len"]
	else:
		tfidf_corpus_size = config["tf-idf_config"]["corpus_size"]
		bm25_corpus_size = config["bm25_config"]["corpus_size"]
		bm25_avg_doc_len = config["bm25_config"]["avg_doc_len"]
	
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"

	###################################################################
	# INITIALIZE SEARCH ENGINES
	###################################################################
	search_1_init_start = time.perf_counter()
	tf_idf = TF_IDF(
		bow_dir,
		depth=args.max_depth, 
		corpus_size=tfidf_corpus_size,
		use_json=args.use_json
	)
	search_1_init_end = time.perf_counter()
	search_1_init_elapsed = search_1_init_end - search_1_init_start
	print(f"Time to initialize TF-IDF search: {search_1_init_elapsed:.6f} seconds")

	search_2_init_start = time.perf_counter()
	bm25 = BM25(
		bow_dir, 
		depth=args.max_depth,
		corpus_size=bm25_corpus_size, 
		avg_doc_len=bm25_avg_doc_len,
		use_json=args.use_json
	)
	search_2_init_end = time.perf_counter()
	search_2_init_elapsed = search_2_init_end - search_2_init_start
	print(f"Time to initialize BM25 search: {search_2_init_elapsed:.6f} seconds")
	
	search_3_init_start = time.perf_counter()
	vector_search = VectorSearch(
		index_dir,
		args.max_depth,
		device
	)
	search_3_init_end = time.perf_counter()
	search_3_init_elapsed = search_3_init_end - search_3_init_start
	print(f"Time to initialize Vector search: {search_3_init_elapsed:.6f} seconds")

	search_4_init_start = time.perf_counter()
	rerank = ReRankSearch(
		bow_dir, 
		index_dir, 
		corpus_size=bm25_corpus_size,
		avg_doc_len=bm25_avg_doc_len,
		device=device,
		use_tf_idf=False,
		use_json=args.use_json
	)
	search_4_init_end = time.perf_counter()
	search_4_init_elapsed = search_4_init_end - search_4_init_start
	print(f"Time to initialize Rerank search: {search_4_init_elapsed:.6f} seconds")

	search_engines = [
		("tf-idf", tf_idf), 
		("bm25", bm25), 
		("vector", vector_search),
		("rerank", rerank)
	]

	###################################################################
	# EXACT PASSAGE RECALL
	###################################################################
	# Sample passages from data in the dataset for exact passage 
	# matching and recall.
	random.seed(4321)
	path = "./InvestopediaDownload/graph/"
	if args.max_depth > 1:
		file_path = os.path.join(
			path, f"expanded_article_map_depth{args.max_depth}.json"
		)
	else:
		file_path = os.path.join(path, "article_map.json")
	
	with open(file_path, "r") as f:
		data = json.load(f)
		sampled_files = random.sample(list(data.keys()), 5)	# sample size of 5 files.
	
	# Process sample files paths.
	sampled_files = [
		os.path.join("./InvestopediaDownload", data[file]["path"])
		for file in sampled_files
	] # Build path.
	sampled_files = [
		file.replace("./data/", "data/") for file in sampled_files
	] # Replace "./data/" from path with just "data/".
	sampled_files = [
		file for file in sampled_files if os.path.exists(file)
	] # Remove files that do not exist.

	# Sample text passages from each file.
	file_passages = list()
	for file in sampled_files:
		with open(file, "r") as f:
			soup = BeautifulSoup(f.read(), "lxml")
		
		try:
			file_text = process_page(soup)
		except:
			print(f"Unable to process text from file {file}. Skipping file.")
			continue

		text_tokens = word_tokenize(file_text)
		ceiling = math.ceil(len(text_tokens) * 0.75)	# 75% of text
		text_length = 25								# number of tokens
		start_index = random.randint(0, ceiling)
		text_passage = " ".join(
			text_tokens[start_index: start_index + text_length]
		)												# actual passage compiled
		file_passages.append((file, text_passage))
	
	# Given passages that are directly pulled from random articles, 
	# determine if the passage each search engine retrieves is correct.
	query_passages = file_passages
	print("=" * 72)

	# Iterate through each search engine (sparse vector search engines 
	# like BM25 or TF-IDF).
	for name, engine in search_engines:#[:2]:
		# Skip vector search because it is not designed/optimized for 
		# scaling (even more so at inference time).
		# if name == "vector":
		# 	continue

		# Search engine banner text.
		print(f"Searching with {name}")
		search_times = []

		# Iterate through each passage and run the search with the 
		# search engine.
		for file, query in query_passages:
			# Run the search and track the time it takes to run the 
			# search.
			query_search_start = time.perf_counter()
			results = engine.search(query)
			query_search_end = time.perf_counter()
			query_search_elapsed = query_search_end - query_search_start

			# Print out the search time and the search results.
			print(f"Search returned in {query_search_elapsed:.6f} seconds")
			print()
			print_results(results, search_type=name, print_doc=args.print_docs)

			# Get top-k accuracy.
			found_index = -1
			for idx, result in enumerate(results):
				if file == result[1]:
					found_index = idx + 1
			
			if found_index < 0:
				print(f"Target article was not found in search (top-50)")
			else:
				print(f"Target article was found in search (top-50) in position: {found_index}")
			
			for k in [5, 10, 25, 50]:
				print(f"top-{k}: {found_index <= k and found_index > -1}")

			# Append the search time to a list.
			search_times.append(query_search_elapsed)

		# Compute and print the average search time.
		assert len(search_times) != 0, \
			"Expected there to be sufficient queries to the search engines. Recieved 0."
		avg_search_time = sum(search_times) / len(search_times)
		print(f"Average search time: {avg_search_time:.6f} seconds")
		print("=" * 72)

	###################################################################
	# GENERAL QUERY
	###################################################################
	# Given passages that have some relative connection to random 
	# articles, determine if the passage each search engine retrieves 
	# is correct.
	file_passages = list()
	for file in sampled_files:
		with open(file, "r") as f:
			soup = BeautifulSoup(f.read(), "lxml")
		
		try:
			file_text = process_page(soup)
		except:
			print(f"Unable to process text from file {file}. Skipping file.")
			continue

		# Split text and remove "empty" strings.
		split_text = file_text.split("\n\n")
		while "\n" in split_text or "" in split_text:
			if "\n" in split_text:
				split_text.remove("\n")
			
			if "" in split_text:
				split_text.remove("")

		# Sample from the text splits using the text lengths as the 
		# sampling weights.
		text = random.choices(
			split_text,
			weights=[len(text) for text in split_text]	
		)

		file_passages.append((file, build_query(text, device)))
		# file_passages.append((file, build_query(text)))
	
	for name, engine in search_engines:
		# Search engine banner text.
		print(f"Searching with {name}")
		search_times = []

		# Iterate through each passage and run the search with the 
		# search engine.
		for file, query in query_passages:
			# Run the search and track the time it takes to run the 
			# search.
			query_search_start = time.perf_counter()
			results = engine.search(query)
			query_search_end = time.perf_counter()
			query_search_elapsed = query_search_end - query_search_start

			# Print out the search time and the search results.
			print(f"Search returned in {query_search_elapsed:.6f} seconds")
			print()
			print_results(results, search_type=name, print_doc=args.print_docs)

			# Get top-k accuracy.
			found_index = -1
			for idx, result in enumerate(results):
				if file == result[1]:
					found_index = idx + 1
			
			if found_index < 0:
				print(f"Target article was not found in search (top-50)")
			else:
				print(f"Target article was found in search (top-50) in position: {found_index}")
			
			for k in [5, 10, 25, 50]:
				print(f"top-{k}: {found_index <= k and found_index > -1}")

			# Append the search time to a list.
			search_times.append(query_search_elapsed)

		# Compute and print the average search time.
		assert len(search_times) != 0, \
			"Expected there to be sufficient queries to the search engines. Recieved 0."
		avg_search_time = sum(search_times) / len(search_times)
		print(f"Average search time: {avg_search_time:.6f} seconds")
		print("=" * 72)
	
	return


def search_loop() -> None:
	'''
	Run an infinite loop (or until the exit phrase is specified) to
		perform search on wikipedia.
	@param: takes no arguments.
	@return: returns nothing.
	'''
	# Read in the title text (ascii art).
	with open("title.txt", "r") as f:
		title = f.read()

	exit_phrase = "Exit Search"
	print(title)
	print()
	search_query = input("> ")
	return 


def main() -> None:
	'''
	Main method. Will either run search engine tests or interactive
		search depending on the program arguments.
	@param: takes no arguments.
	@return: returns nothing.
	'''
	# Set up argument parser.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--test",
		action="store_true",
		help="Specify whether to run the search engine tests. Default is false/not specified."
	)
	parser.add_argument(
		"--use_json",
		action="store_true",
		help="Whether to read from JSON or msgpack files. Default is false/not specified."
	)
	# parser.add_argument(
	# 	"--num_proc",
	# 	type=int,
	# 	default=1,
	# 	help="How many processors to use. Default is 1."
	# )
	# parser.add_argument(
	# 	"--num_thread",
	# 	type=int,
	# 	default=1,
	# 	help="How many threads to use. Default is 1."
	# )
	parser.add_argument(
		"--max_depth",
		type=int,
		default=1,
		help="How deep should the graph traversal go across links. Default is 1/not specified."
	)
	parser.add_argument(
		"--print_docs",
		action="store_true",
		help="Whether to print out the entire document text when printing the results. Default is false/not specified."
	)
	args = parser.parse_args()

	# Depending on the arguments, either run the search tests or just
	# use the general search function.
	if args.test:
		test(args)
	else:
		search_loop()

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()