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
import time

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import torch

from preprocess import process_page
from search import TF_IDF, BM25#, VectorSearch, ReRankSearch
from search import print_results


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
	index_dir = "./test-temp"


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
	
	model = "bert"
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
		corpus_size=tfidf_corpus_size
	)
	search_1_init_end = time.perf_counter()
	search_1_init_elapsed = search_1_init_end - search_1_init_start
	print(f"Time to initialize TF-IDF search: {search_1_init_elapsed:.6f} seconds")

	search_2_init_start = time.perf_counter()
	bm25 = BM25(
		bow_dir, 
		depth=args.max_depth,
		corpus_size=bm25_corpus_size, 
		avg_doc_len=bm25_avg_doc_len
	)
	search_2_init_end = time.perf_counter()
	search_2_init_elapsed = search_2_init_end - search_2_init_start
	print(f"Time to initialize BM25 search: {search_2_init_elapsed:.6f} seconds")
	
	# search_3_init_start = time.perf_counter()
	# vector_search = VectorSearch()
	# search_3_init_end = time.perf_counter()
	# search_3_init_elapsed = search_3_init_end - search_3_init_start
	# print(f"Time to initialize Vector search: {search_3_init_elapsed:.6f} seconds")

	# search_4_init_start = time.perf_counter()
	# rerank = ReRankSearch(bow_dir, index_dir, model, device=device)
	# search_4_init_end = time.perf_counter()
	# search_4_init_elapsed = search_4_init_end - search_4_init_start
	# print(f"Time to initialize Rerank search: {search_4_init_elapsed:.6f} seconds")

	search_engines = [
		("tf-idf", tf_idf), 
		# ("bm25", bm25), 
		# ("vector", vector_search),
		# ("rerank", rerank)
	]

	###################################################################
	# EXACT PASSAGE RECALL
	###################################################################
	# Sample passages from data in the dataset for exact passage 
	# matching and recall.
	random.seed(1234)
	path = "./InvestopediaDownload/graph/"
	if args.max_depth > 1:
		file_path = os.path.join(
			path, f"term_article_graph_depth{args.max_depth}.json"
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
	] # Remove extra "./data/" from path.
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
	# query_passages = [
	# 	passage for _, passage in file_passages
	# ]
	query_passages = file_passages
	print("=" * 72)

	# Iterate through each search engine.
	for name, engine in search_engines:
		# Skip vector search because it is not designed/optimized for 
		# scaling (even more so at inference time).
		if name == "vector":
			continue

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
			# print_results(results, search_type=name)

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
		exit()

	###################################################################
	# GENERAL QUERY
	###################################################################
	# Given passages that have some relative connection to random 
	# articles, determine if the passage each search engine retrieves 
	# is correct.
	query_text = [

	]
	
	for name, engine in search_engines:
		pass
	pass


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
	pass


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