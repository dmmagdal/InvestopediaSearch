# Investopedia Search

Description: Build a search engine as part of a larger RAG project that looks at key concepts and terms (as well as a few additional articles) from Investopedia.


### Setup


### Notes

 - It is NOT recommended using a `max_depth` higher than `3` 
 - Sparse vector representation does not require too many files because the number of files in this dataset.
     - Means that we can do away with most extraneous metadata folders for bag of words (ie `idf cache` or `redirect_cache`).
     - Because there are not many documents, we won't need to look at "file-based" inverted index unlike in WikipediaSearch.
 - Inverted index support
     - document-based:
     - trie-based:
     - category tree-based:


### TODO

[ ] Complete `setup.py`
[ ] Validate TF-IDF on exact passage recall
[ ] Validate BM25 on exact passage recall
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