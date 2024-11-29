import argparse
import re
import json
import gzip
import zipfile
from glob import iglob
import os.path
import lxml.etree as et
import logging
import unicodedata
import csv

logger = logging.getLogger("marshall_perseus")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--perseus_root", dest="perseus_root", help="Root of Perseus data repos")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--tsv_output", dest="tsv_output", help="Sheet of authors and their works")
    args = parser.parse_args()

    authors = {}
    titles = {}
    author_to_titles = {}
    
    parser = et.XMLParser(recover=True)

    with gzip.open(args.output, "wt") as ofd:
        for fname in iglob("{}/**/*.xml".format(args.perseus_root), recursive=True):
            with open(fname, "rt") as ifd:
                xml = et.parse(ifd, parser=parser)
                keep = None
                title = None
                author = None
                for ts in xml.iter("{*}titleStmt"):
                    for t in ts.iter("{*}title"):
                        if t.text:
                            title = t.text


                    for a in ts.iter("{*}author"):
                        if a.text and not re.match(r"^\s*$", a.text):
                            author = a.text
                title = re.sub(r"\s+", " ", title) if title else "?"
                author = re.sub(r"\s+", " ", author) if author else "?"

                content = []
                for div in xml.iter("{*}div"):
                    for k, v in div.attrib.items():
                        if v == "grc" and div.attrib.get("type") != "translation":
                            cs = []
                            for c in re.sub(r"\s+", " ", " ".join(div.xpath(".//text()"))):
                                if unicodedata.name(c, "UNKNOWN").split()[0] in ["SPACE", "GREEK"]: 
                                    cs.append(c)
                            txt = re.sub(r"\s+", " ", "".join(cs))
                            if not re.match(r"^\s*$", txt):
                                content.append(txt.split())
                if len(content) > 0:
                    author_to_titles[author] = author_to_titles.get(author, set())
                    author_to_titles[author].add(title)
                    ofd.write(
                        json.dumps(
                            {
                                "file" : fname,
                                "author" : author,
                                "title" : title,
                                "content" : content
                            }
                        ) + "\n"
                    )
    
    with open(args.tsv_output, "wt") as ofd:
        c = csv.DictWriter(ofd, fieldnames=["author", "title", "term_post_quem", "term_ante_quem", "comment"], delimiter="\t")
        c.writeheader()
        for author, titles in author_to_titles.items():
            c.writerow({"author" : author})
            for title in titles:
                c.writerow({"title" : title})                
        
