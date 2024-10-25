#!/usr/bin/python

import argparse
import sys
import json

from .main import *


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
    parser = argparse.ArgumentParser(description='Wrapper around Spacy and Trankit.')
    parser.add_argument('text', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="file with text to process (default: stdin)")
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

    for n_doc, doc in enumerate(nlp.pipe(text_reader(args.text, args.lines))):

        if args.tree:
            docs_for_displacy.append(doc)

        for n_sent, sentence in enumerate(doc.sents):
            for n_token, token in enumerate(sentence):
                print(token_to_str(token, args.info, index=f'{n_doc}.{n_sent}.{n_token}' if args.id else None))

        if args.sep:
            print()

    if docs_for_displacy:
        render_parse_tree_html(docs_for_displacy)


def sentencize_cli():
    """
    Process with spaCy, printing resulting sentences.
    """
    parser = make_base_arg_parser()
    # parser.add_argument('--offsets', action='store_true', help="whether to include token/sentence start/end characters")
    parser.add_argument('--tree', action='store_true', help="whether to render dependency tree with displacy")
    parser.add_argument('--json', action='store_true', help="whether to print sentences as json(lines) format")
    parser.add_argument('--sep', action='store_true', help="whether to separate sentences for different docs with (double) newlines")

    parser.add_argument('--spans', action='store_true', help="Whether to output lines like {start: ..., end: ..., text: ...}.")
    parser.add_argument('--context', action='store_true', help="whether to prepend sentences with some context.")
    parser.add_argument('--chunks', action='store_true', help="whether to prepend sentences with some context.")
    parser.add_argument('--min_sent', type=int, default=1, help="each chunk is at least this many sentences (only with --context or --chunks; >=1).")
    parser.add_argument('--min_tokens', type=int, default=15, help="each chunk has at least this many tokens (only with --context or --chunks).")
    parser.add_argument('--max_tokens', type=int, default=50, help="each chunk is at most this many tokens, except if necessary for min_sent/min_tokens (only with --context or --chunks).")
    args = parser.parse_args()

    if args.context and args.chunks:
        logging.warning('Cannot do both --context and --chunked; ignoring the latter.')
        args.chunks = False

    docs_for_displacy = []
    nlp = load_trankit_model(args.lang) if args.trf else load_spacy_model(args.lang)

    sentencizer = sentencize_contextual if args.context else sentencize_chunked if args.chunks else sentencize
    sentencizer = functools.partial(sentencizer, language=args.lang, use_trf=args.trf, return_spacy=True)
    if args.context or args.chunks:
        sentencizer = functools.partial(sentencizer, min_n_sent=args.min_sent, min_n_tokens=args.min_tokens, max_n_tokens=args.max_tokens)

    for n_doc, doc in enumerate(nlp.pipe(text_reader(args.text, args.lines))):

        for n_sent, sent in enumerate(sentencizer(doc)):

            if args.tree or args.json:
                sent_as_doc = sent.as_doc()

            if args.tree:
                docs_for_displacy.append(sent_as_doc)

            if args.json:
                if args.id:
                    sent_as_doc['id'] = f'{n_doc}.{n_sent}'
                s = json.dumps(sent_as_doc.to_json())
            elif args.spans:
                s = json.dumps({'start': sent.start, 'end': sent.end, 'text': sent.text})
            else:
                s = sentence_to_str(sent, index=f'{n_doc}.{n_sent}' if args.id else None)

            print(s)

        if args.sep:
            print()

    if docs_for_displacy:
        render_parse_tree_html(docs_for_displacy)


def spacy_cli():
    """
    Process with spaCy, printing (each) resulting doc as json.
    """
    parser = make_base_arg_parser()
    parser.add_argument('--tree', action='store_true', help="whether to render dependency tree with displacy")
    args = parser.parse_args()

    nlp = load_trankit_model(args.lang) if args.trf else load_spacy_model(args.lang)

    docs_for_displacy = []

    for n_doc, doc in enumerate(nlp.pipe(text_reader(args.text, args.lines))):
        if args.tree:
            docs_for_displacy.append(doc)

        d = doc.to_json()
        if args.id:
            d['id'] = n_doc
        s = json.dumps(d)
        print(s)

    if docs_for_displacy:
        render_parse_tree_html(docs_for_displacy)


def text_reader(source, linewise: bool):
    texts = source if linewise else [source.read()]
    for text in texts:
        yield text.rstrip()


