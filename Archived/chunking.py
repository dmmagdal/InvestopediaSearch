# chunking.py
# Experiment with how to tokenize, chunk, and pad long texts 
# Python 3.11
# Windows/MacOS/Linux


import json
import os
import random
from typing import Dict, List

from bs4 import BeautifulSoup
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import BatchEncoding

from preprocess import load_model, process_page


def get_chunk_indices(seq_length, chunk_size, overlap_tokens):
    step = chunk_size - overlap_tokens
    indices = torch.arange(0, seq_length, step)
    chunks = [(start, min(start + chunk_size, seq_length)) for start in indices]
    return torch.tensor(chunks)


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
	assert isinstance(article_text, str),\
		f"Required argument 'article_text' expected a str, recieved {type(article_text)}."

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

	split_text = [" ".join(split_text)]

	all_tokens = list()
	for text_chunk in split_text:
		tokens = tokenizer(
			text_chunk, 
			return_offsets_mapping=True,
			return_tensors="pt" # required in order to pass through to model.
		)
		print(tokens)
		# all_tokens.append(tokens)
		for key, value in tokens.items():
			print(f"{key} shape: {value.shape}")

		# Verify the batch dim of the tensors is 1 (passing in only 1 
		# input string to the tokenizer should result in all tensors 
		# having a batch size of 1).
		assert all([value.size(0) == 1 for value in tokens.values()]),\
			"Expected all tensors in tokenized output to have batch size of 1."
		assert all(tensor.size(1) == next(iter(tokens.values())).size(1) for tensor in tokens.values()),\
			"Expected all length dimensions in tokenized output tensors to be uniform."

		# Given the tensor length, the context length, and token 
		# overlap, compute the maximum length needed to pad the tensors
		# in order for the chunking to work evenly.
		tensor_length = list(tokens.values())[0].size(1)
		step_size = context_length - overlap
		chunks_needed = (tensor_length + step_size - 1) // step_size
		max_pad_length = (chunks_needed * step_size) + overlap
		print(max_pad_length)

		all_tokens_chunks = dict()
		for key, tensor in tokens.items():
			if tensor_length < context_length:
				# Pad each tensor to be divisible by the chunking 
				# process.
				pad_size = max_pad_length - tensor_length
				if len(tensor.shape) > 2:
					# tokens[key] = F.pad(tensor, (0, 0, 0, pad_size))
					chunk = F.pad(tensor, (0, 0, 0, pad_size))
				else:
					# tokens[key] = F.pad(tensor, (0, pad_size))
					chunk = F.pad(tensor, (0, pad_size))
				
				# Assert that the chunk length matches the context 
				# length. It should given that we already padded the
				# tensors first.
				assert chunk.size(1) == context_length, \
					f"Mismatched chunked tensor length. Expected {context_length}, received {chunk.size(1)}"
				
				# Append chunk to the list of tensor chunks.
				tensor_chunks = [chunk]
			else:
				# Chunk each tensor given the context length and token 
				# overlap.
				# start_indices = torch.arange(0, tensor_length, step_size)
				start_indices = list(range(0, tensor_length, step_size))
				tensor_chunks = list()
				for start_idx in start_indices:
					end_idx = min(
						start_idx + context_length, tensor_length
					)
					# chunk = padded_tensor.narrow(
					# 	1, start_idx, end_idx - start_idx
					# )
					chunk = tensor.narrow(
						1, start_idx, end_idx - start_idx
					)

					pad_size = context_length - chunk.size(1)
					if len(tensor.shape) > 2:
						chunk = F.pad(chunk, (0, 0, 0, pad_size))
					else:
						chunk = F.pad(chunk, (0, pad_size))

					# Assert that the chunk length matches the context 
					# length. It should given that we already padded the
					# tensors first.
					assert chunk.size(1) == context_length, \
						f"Mismatched chunked tensor length. Expected {context_length}, received {chunk.size(1)}"
					
					# Append chunk to the list of tensor chunks.
					tensor_chunks.append(chunk)

			# Store chunks to dictionary.
			all_tokens_chunks[key] = tensor_chunks

		# Validate shapes.
		for key, value in tokens.items():
			print(f"{key} shape: {value.shape}")
		
		for key, value in all_tokens_chunks.items():
			print(f"{key} shapes:")
			output_strings = [f"\t{tensor.shape}" for tensor in value]
			print("\n".join(output_strings))

		values = all_tokens_chunks["input_ids"]
		print(values[0])
		print(value[-1])

		assert all(len(values) == len(next(iter(all_tokens_chunks.values()))) for values in all_tokens_chunks.values()),\
			"Expected the number of chunks for tokenized data to be uniform across all keys."

		token_chunks_list = list()
		keys = list(tokens.keys())
		for idx in range(len(values)):
			token_chunks_list.append({
				key: all_tokens_chunks[key][idx] for key in keys
			})
		return token_chunks_list
		exit()

		###############################################################
		# TEST 1:
		###############################################################
		# # Tokenize.
		# tokens = tokenizer(
		# 	text_chunk, 
		# 	# return_attention_mask=True,
		# 	return_offsets_mapping=True,
		# 	return_tensors="pt",
		# 	# padding="max_length",
		# )
		# print(tokens)
		# # all_tokens.append(tokens)

		# input_ids = tokens["input_ids"]
		# token_type_ids = tokens["token_type_ids"]
		# attention_mask = tokens["attention_mask"]
		# offset_mapping = tokens["offset_mapping"]
		# pad_token_id = 0

		# start_idx = 0

		# while start_idx < len(input_ids):
		# 	end_idx = min(start_idx + context_length, len(input_ids))
			
		# 	# Extract data from the current chunk
		# 	chunk_input_ids = input_ids[start_idx:end_idx]
		# 	chunk_type_ids = token_type_ids[start_idx:end_idx]
		# 	chunk_attention_mask = attention_mask[start_idx:end_idx]
		# 	chunk_offset_mappings = offset_mapping[start_idx:end_idx]
			
		# 	# Determine the start and end character indices in the 
		# 	# original text.
		# 	mapping = chunk_offset_mappings.squeeze(0)
		# 	mini_start_idx = 0
		# 	while mini_start_idx < mapping.size(0) and torch.equal(mapping[mini_start_idx], torch.tensor([0, 0])):
		# 		mini_start_idx += 1

		# 	mini_end_idx = mapping.size(0)
		# 	while mini_start_idx < mini_start_idx and torch.equal(mapping[mini_end_idx], torch.tensor([0, 0])):
		# 		mini_start_idx -= 1

		# 	chunk_start = mapping[0][0]
		# 	chunk_end = mapping[-1][-1]
		# 	# chunk_start = chunk_offset_mappings[0][0]
		# 	# chunk_end = chunk_offset_mappings[-1][-1]

		# 	# Pad the last chunk if necessary
		# 	if len(chunk_input_ids) < context_length:
		# 		# chunk_input_ids += [pad_token_id] * (context_length - len(chunk_input_ids))
		# 		pad_tensor = torch.tensor([pad_token_id] * (context_length - len(chunk_input_ids)))
		# 		pad_tensor = pad_tensor.unsqueeze(0)
		# 		chunk_input_ids = torch.cat((chunk_input_ids, pad_tensor), dim=1)
			
		# 	all_tokens.append(
		# 		{	
		# 			"input_ids": chunk_input_ids, 
		# 			"attention_mask": chunk_attention_mask,
		# 			"chunk_type_ids": chunk_type_ids,
		# 			"text_range": (chunk_start, chunk_end)
		# 		}
		# 	)

		# 	# MOve the start index forward, considering the overlap.
		# 	start_idx = end_idx - overlap  # Move forward, keeping the overlap

		# all_tokens.append(tokens)
		
	return all_tokens


def main():
	# Input values to search engines.
	with open("../config.json", "r") as f:
		config = json.load(f)

	path = "../InvestopediaDownload/graph/"
	file_path = os.path.join(path, "article_map.json")
	random.seed(1234)
	
	with open(file_path, "r") as f:
		data = json.load(f)
		sampled_files = random.sample(list(data.keys()), 5)	# sample size of 5 files.
	
	# Process sample files paths.
	sampled_files = [
		os.path.join("../InvestopediaDownload", data[file]["path"])
		for file in sampled_files
	] # Build path.
	sampled_files = [
		file.replace("./data/", "data/") for file in sampled_files
	] # Replace "./data/" from path with just "data/".
	sampled_files = [
		file for file in sampled_files if os.path.exists(file)
	] # Remove files that do not exist.

	file_passages = list()
	for file in sampled_files:
		with open(file, "r") as f:
			soup = BeautifulSoup(f.read(), "lxml")
		
		try:
			file_text = process_page(soup)
		except:
			print(f"Unable to process text from file {file}. Skipping file.")
			continue

		tokenizer, model = load_model(config)
		# model = model.to(device)
		all_tokens = vector_preprocessing(file_text, config, tokenizer)
		token_keys = ["input_ids", "attention_mask", "chunk_type_ids"]
		for tokens in all_tokens:
			token_details = {
				key: value for key, value in tokens.items()
				if key in token_keys
			}
			# model(tokens)
			# model(**tokens)
			model(**token_details)
			# model(**BatchEncoding(token_details).to(device))
			# model(**BatchEncoding(token_details))

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()