import argparse
import re
import json
import gzip
import zipfile
from glob import iglob, glob
import os.path
import lxml.etree as et
import logging
import unicodedata
import csv
import sys
import pandas
import numpy


logger = logging.getLogger("marshall_data")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--perseus_root", dest="perseus_root", help="Root of Perseus data repos")
    parser.add_argument("--iowa_spreadsheet", dest="iowa_spreadsheet", help="Spreadsheet of dates")
    parser.add_argument("--iowa_nt", dest="iowa_nt", help="New Testament apocrypha")
    parser.add_argument("--iowa_ot", dest="iowa_ot", help="Old Testament apocrypha")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    dates = {}
    df = pandas.read_excel(args.iowa_spreadsheet)
    for row in df.iloc:
        if row["Century"] != "Exclude":
            taq = row["Terminus Ante Quem"]
            tpq = taq if row["Terminus Post Quem"] == "??" else row["Terminus Post Quem"]
            date = (taq + tpq) / 2.0
            if isinstance(row["Work Urn"], str):
                dates[row["Work Urn"].split(":")[-1]] = date
            else:
                dates[(row["Text Group"], row["Work"])] = date

    
    authors = {}
    titles = {}
    author_to_titles = {}
    
    parser = et.XMLParser(recover=True)

    with gzip.open(args.output, "wt") as ofd:
        for fname in iglob("{}/**/*.xml".format(args.perseus_root), recursive=True):
            rest, _ = os.path.split(fname)
            rest, compB = os.path.split(rest)
            rest, compA = os.path.split(rest)
            urn = "{}.{}".format(compA, compB)
            if urn not in dates:
                continue
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
                                content.append(txt)
                if len(content) > 0:
                    author_to_titles[author] = author_to_titles.get(author, set())
                    author_to_titles[author].add(title)
                    ofd.write(
                        json.dumps(
                            {
                                "file" : fname,
                                "author" : author,
                                "title" : title,
                                "year" : dates[urn],
                                "content" : "\n".join(content).strip()
                            }
                        ) + "\n"
                    )
