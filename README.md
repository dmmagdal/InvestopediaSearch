# Investopedia Search

Description: Build a search engine as part of a larger RAG project that looks at key concepts and terms (as well as a few additional articles) from Investopedia.


### Setup


### Notes

 - It is NOT recommended using a `max_depth` higher than `3` 
     - The time it takes to scrape with a max_depth of `3` or larger is not worth anyone's time.
 - Sparse vector representation does not require too many files because the number of files in this dataset.
     - Means that we can do away with most extraneous metadata folders for bag of words (ie `idf cache` or `redirect_cache`).
     - Because there are not many documents, we won't need to look at "file-based" inverted index unlike in WikipediaSearch.
 - Inverted index support
     - document-based:
     - trie-based:
     - category tree-based:
 - Ran `preprocess.py` on server with P100 card (16 GB VRAM).
     - Command used: `python preprocessing --max_depth 2 --num_proc 16 --batch_size 8 --override_gpu2cpu_limit --max_files_per_chunk 500`
         - Runs both bag of words and vector preprocessing.
         - Overrides the default GPU auto toggle to CPU limit for number of processors (limit is 4).
         - Sets the number of processors to 16.
         - Sets the batch size to 8 for the embedding model.
         - The `max_depth` is for the graph mapping all articles. Set to 2 above but results are similar for when set to 1.
         - The `max_files_per_chunk` is for the maximum number of files to contain within a chunk when chunking the files for vector processing. Using the value of 500 gave us the below memory footprint but a lower should give a lower footprint.
     - Memory: 27 GB RAM
     - VARM: ~10 GB
     - Time: ~3 hours 45 minutes


### TODO

[ ] Complete `setup.py`
[X] Validate TF-IDF on exact passage recall
[X] Validate BM25 on exact passage recall
[ ] Validate ReRank on exact passage recall
[ ] Validate all search on more direct questions
[ ] Full RAG with small LLMs


### STRETCH TODO

[ ] Graph DB (Graph RAG, FastRAG)
[ ] Graph relation between articles? 
[ ] Category tree inverted index
[ ] Rust implementation
[ ] JS implementation
[ ] Combine with WikipediaSearch???