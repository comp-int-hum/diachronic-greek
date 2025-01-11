import sys
import logging
import gzip
import math
import json
import argparse
import numpy as np
import numpy
from sklearn.metrics.pairwise import cosine_similarity
#from detm import DETM
import pandas
#import torch
from gensim.models import Word2Vec
import numpy as np
import numpy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



logger = logging.getLogger("generate_word_similarity_table")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--embeddings", dest="embeddings", help="W2V embeddings file")
    parser.add_argument("--output", dest="output", help="File to save table")
    parser.add_argument("--top_neighbors", dest="top_neighbors", default=10, type=int, help="How many neighbors to return")
    parser.add_argument('--target_words', default=[], nargs="*", help='Words to consider')
    parser.add_argument("--language_code", default=None)
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model, tokenizer = None, None
    if args.language_code:
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-1.3B", src_lang=args.language_code
        )
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")
        
    w2v = Word2Vec.load(args.embeddings)

    neighbors = []
    for w in args.target_words:
        row = [w]
        for ow, op in w2v.wv.most_similar(w, topn=args.top_neighbors):
            row.append("{}:{:.02f}".format(ow, op))
        if model:
            for i in range(len(row)):
                inputs = tokenizer(row[i], return_tensors="pt")
                translated_tokens = model.generate(
                    **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"), max_length=10
                )
                tr = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                row[i] = "{}({})".format(row[i], tr)
        neighbors.append(row)
        
    pd = pandas.DataFrame(neighbors)
    with open(args.output, "wt") as ofd:
        ofd.write(pd.to_latex(index_names=False, index=False, header=False))
