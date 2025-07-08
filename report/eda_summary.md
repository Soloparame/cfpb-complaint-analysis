## Task 1: EDA Summary

The CFPB complaint dataset consists of a wide range of consumer financial complaints, including products such as Credit Cards, BNPL, Savings Accounts, and more.

From the initial EDA, we found that the most frequent complaint product is "Credit card", followed by "Money transfer..." and "Buy Now, Pay Later". Out of the full dataset, around XX% of entries had a consumer complaint narrative, while the remaining records did not include detailed text.

We analyzed the word count distribution of narratives and observed a wide range—from very short narratives (less than 10 words) to very long ones exceeding 500 words. This indicates a need for standard preprocessing. We filtered the dataset to include only five targeted financial products and cleaned the narrative text using standard NLP techniques such as lowercasing and regex-based cleaning. The cleaned dataset has been saved for downstream tasks in the RAG pipeline.
