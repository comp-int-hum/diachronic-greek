import logging
import gzip
import json
import argparse
import numpy
import torch
from detm import Corpus, DETM, AbstractDETM, apply_model
import sys

logger = logging.getLogger("apply_detm")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", dest="input", help="Data file")
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--output", dest="output", help="File to save model to", required=True)
    parser.add_argument("--max_subdoc_length", dest="max_subdoc_length", type=int, default=200, help="Documents will be split into subdocuments of at most this number of tokens")
    parser.add_argument("--lowercase", dest="lowercase", default=False, action="store_true")
    parser.add_argument("--time_field", dest="time_field", default="time")
    parser.add_argument("--content_field", dest="content_field", default="content")
    parser.add_argument('--device') #, choices=["cpu", "cuda"], help='')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--limit_docs', type=int, help='')
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    if not args.device:
       args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    elif args.device == "cuda" and not torch.cuda.is_available():
       logger.warning("Setting device to CPU because CUDA isn't available")
       args.device = "cpu"

    with gzip.open(args.model, "rb") as ifd:
        model = torch.load(ifd, map_location=torch.device(args.device), weights_only=False)
        
       
    corpus = Corpus()
    with gzip.open(args.input, "rt") as ifd:
        for i, line in enumerate(ifd):
            if args.limit_docs and i >= args.limit_docs:
                break
            corpus.append(json.loads(line))

    subdocs, times = corpus.filter_for_model(model, args.max_subdoc_length, args.content_field, args.time_field, args.lowercase)
            
    model = model.to(args.device)
    ppl = apply_model(
        #corpus,
        # ppl = perplexity_on_corpus(
        #     model,
        model,
        subdocs,
        times,
        args.batch_size
        
        #max_subdoc_length=args.max_subdoc_length,
        #content_field=args.content_field,
        #time_field=args.time_field,
        #lowercase=args.lowercase,
        #device="cuda"
    )
    
    print(ppl)
