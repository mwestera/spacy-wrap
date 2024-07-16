#!/usr/bin/python

import argparse
import sys
import json

from .main import *

from spacy import displacy

# TODO: Where do cached models go?
# TODO: Use deplacy instead of displacy?


"""
Author: Matthijs Westera

Simple command-line interface to spacy and trankit. Allows word tokenization and sentence segmentation, for given language or using 
automatic language detection, as well as printing spacy parses directly as json.

Can print parse information per token and display a parse tree.

Examples:

$ echo "Here's just a short text. For you to parse." | tokenize --info -tree

Or, to process each line from a file separately (and this time using a transformer model, --trf):

$ cat lots_of_dutch_sentences.txt | tokenize --info --trf --lang nl --lines

$ cat texts_in_various_languages.txt | sentencize --trf --lines

Note: In this case, will detect language separately for each input line.

Or output full sentence parses in json format:

$ cat texts_in_various_languages.txt | sentencize --lines --lang nl --json

Or entire spacy docs as json:

$ cat texts_in_various_languages.txt | spacyjson --lines --lang nl --json

"""


def make_base_arg_parser():
    parser = argparse.ArgumentParser('Simple interface to Spacy.')
    parser.add_argument('text', nargs='?', type=str, default=sys.stdin, help="text to process (default: stdin)")
    parser.add_argument('--lang', type=str, default=None, help="language (otherwise: auto-detect)")
    parser.add_argument('-l', '--lines', action='store_true', help="whether to process individual lines from the input")
    parser.add_argument('-t', '--trf', action='store_true', help="whether to use trankit's transformer models instead")
    parser.add_argument('--id', action='store_true', help="whether number docs/sentences/tokens in the output, e.g., 4.3.6.")

    return parser


def tokenize_cli():
    """
    Process with spaCy, printing each token.
    """

    parser = make_base_arg_parser()
    # parser.add_argument('--offsets', action='store_true', help="whether to include token/sentence start/end characters")
    parser.add_argument('--tree', action='store_true', help="whether to render dependency tree with displacy")
    parser.add_argument('--info', action='store_true', help="whether to print parse info alongsize tokens")
    parser.add_argument('--sep', action='store_true', help="whether to separate multiple docs/sentences with (double) newlines.")

    args = parser.parse_args()
    nlp = load_trankit_model(args.lang) if args.trf else load_spacy_model(args.lang)

    docs_for_displacy = []

    for n_text, text in enumerate(text_reader(args.text, args.lines)):
        doc = nlp(text)

        if args.tree:
            docs_for_displacy.append(doc)

        for n_sent, sentence in enumerate(doc.sents):
            for n_token, token in enumerate(sentence):
                print(token_to_str(token, args.info, index=f'{n_text}.{n_sent}.{n_token}' if args.id else None))

            if args.sep:
                print()

        if args.sep:
            print()

    if docs_for_displacy:
        displacy.serve(docs_for_displacy, style="dep")


def sentencize_cli():
    """
    Process with spaCy, printing resulting sentences.
    """
    parser = make_base_arg_parser()
    # parser.add_argument('--offsets', action='store_true', help="whether to include token/sentence start/end characters")
    parser.add_argument('--tree', action='store_true', help="whether to render dependency tree with displacy")
    parser.add_argument('--json', action='store_true', help="whether to print sentences as json(lines) format")
    parser.add_argument('--sep', action='store_true', help="whether to separate multiple docs with (double) newlines.")

    args = parser.parse_args()
    nlp = load_trankit_model(args.lang) if args.trf else load_spacy_model(args.lang)

    docs_for_displacy = []

    for n_text, text in enumerate(text_reader(args.text, args.lines)):
        doc = nlp(text)

        if args.tree:
            docs_for_displacy.append(doc)

        for n_sent, sentence in enumerate(doc.sents):
            if args.json:
                d = sentence.as_doc().to_json()
                if args.id:
                    d['id'] = f'{n_text}.{n_sent}'
                s = json.dumps(d)
            else:
                s = sentence_to_str(sentence, index=f'{n_text}.{n_sent}' if args.id else None)
            print(s)

        if args.sep:
            print()

    if docs_for_displacy:
        displacy.serve(docs_for_displacy, style="dep")


def spacy_cli():
    """
    Process with spaCy, printing (each) resulting doc as json.
    """
    parser = make_base_arg_parser()
    parser.add_argument('--tree', action='store_true', help="whether to render dependency tree with displacy")
    args = parser.parse_args()

    nlp = load_trankit_model(args.lang) if args.trf else load_spacy_model(args.lang)

    docs_for_displacy = []

    for n_text, text in enumerate(text_reader(args.text, args.lines)):
        doc = nlp(text)
        if args.tree:
            docs_for_displacy.append(doc)

        d = doc.to_json()
        if args.id:
            d['id'] = n_text
        s = json.dumps(d)
        print(s)

    if docs_for_displacy:
        displacy.serve(docs_for_displacy, style="dep")


def text_reader(source, linewise: bool):
    if isinstance(source, str):
        texts = source.splitlines() if linewise else [source]
    else:   # it's stdin
        texts = source if linewise else [source.read()]
    for text in texts:
        yield text.rstrip()


