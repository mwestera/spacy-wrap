#!/usr/bin/python

import spacy
import argparse
import sys
import functools
import fasttext
import logging
import json


"""
Author: Matthijs Westera

Simple command-line interface to spacy. Allows word tokenization and sentence segmentation, for given language or using 
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


fasttext.FastText.eprint = lambda x: None   # monkey patch as per https://github.com/facebookresearch/fastText/issues/1067


models = {  # this only in case of spacy; transformer alternatives (--trf) loaded through trankit instead
    'nl': 'nl_core_news_sm',
    'en': 'en_core_web_sm',
}


def make_base_arg_parser():
    parser = argparse.ArgumentParser('Simple interface to Spacy.')
    parser.add_argument('text', nargs='?', type=str, default=sys.stdin, help="text to process (default: stdin)")
    parser.add_argument('--lang', type=str, default=None, help="language (otherwise: auto-detect)")
    parser.add_argument('-l', '--lines', action='store_true', help="whether to process individual lines from the input")
    parser.add_argument('-t', '--trf', action='store_true', help="whether to use trankit's transformer models instead")
    parser.add_argument('--id', action='store_true', help="whether number docs/sentences/tokens in the output, e.g., 4.3.6.")

    return parser


def tokenize():
    """
    Process with spaCy, printing each token.
    """

    parser = make_base_arg_parser()
    # parser.add_argument('--offsets', action='store_true', help="whether to include token/sentence start/end characters")
    parser.add_argument('--tree', action='store_true', help="whether to render dependency tree with displacy")
    parser.add_argument('--info', action='store_true', help="whether to print parse info alongsize tokens")
    parser.add_argument('--sep', action='store_true', help="whether to separate multiple docs/sentences with (double) newlines.")

    args = parser.parse_args()

    texts = text_reader(args.text, args.lines)
    for n_doc, doc in enumerate(process_texts(texts, args.lang, args.trf)):
        for n_sent, sentence in enumerate(doc.sents):
            for n_token, token in sentence:
                print(token_to_str(token, args.info, index=f'{n_doc}.{n_sent}.{n_token}' if args.id else None))

            if args.sep:
                print()

        if args.sep:
            print()

        if args.tree:
            spacy.displacy.serve(doc, style="dep")


def sentencize():
    """
    Process with spaCy, printing resulting sentences.
    """
    parser = make_base_arg_parser()
    # parser.add_argument('--offsets', action='store_true', help="whether to include token/sentence start/end characters")
    parser.add_argument('--tree', action='store_true', help="whether to render dependency tree with displacy")
    parser.add_argument('--json', action='store_true', help="whether to print sentences as json(lines) format")
    parser.add_argument('--sep', action='store_true', help="whether to separate multiple docs with (double) newlines.")

    args = parser.parse_args()

    texts = text_reader(args.text, args.lines)
    for n_doc, doc in enumerate(process_texts(texts, args.lang, args.trf)):
        for n_sent, sentence in enumerate(doc.sents):
            if args.json:
                d = sentence.as_doc().to_json()
                if args.id:
                    d['id'] = f'{n_doc}.{n_sent}'
                s = json.dumps(d)
            else:
                s = sentence_to_str(sentence, index=f'{n_doc}.{n_sent}' if args.id else None)
            print(s)

        if args.sep:
            print()

        if args.tree:
            spacy.displacy.serve(doc, style="dep")


def spacyjson():
    """
    Process with spaCy, printing (each) resulting doc as json.
    """
    parser = make_base_arg_parser()
    args = parser.parse_args()

    texts = text_reader(args.text, args.lines)
    for n_doc, doc in enumerate(process_texts(texts, args.lang, args.trf)):
        d = doc.to_json()
        if args.id:
            d['id'] = n_doc
        s = json.dumps(d)
        print(s)


def text_reader(source, linewise: bool):
    if isinstance(source, str):
        texts = source.splitlines() if linewise else [source]
    else:   # it's stdin
        texts = source if linewise else [source.read()]
    for text in texts:
        yield text.rstrip()


def process_texts(text_reader, lang: str, trf: bool):
    # args = parse_args()

    if lang:
        lang_detector = None
    else:
        import spacy_fastlang   # registers the language_detector
        lang_detector = spacy.blank('en')
        lang_detector.add_pipe("language_detector")

    for text in text_reader:
        if lang_detector:
            doc_detected = lang_detector(text)
            lang = doc_detected._.language
            if doc_detected._.language_score < .5:
                logging.warning(f'Language detected with uncertainty: {lang}={doc_detected._.language_score}: {text}')

        nlp = load_model(lang, trf)
        doc = nlp(text)
        yield doc


def token_to_str(token, verbose: bool, offsets=False, index=None):
    if verbose:
        s = f'{token.idx:<4} {token.idx + len(token.text):<4}    {token.text + ("("+token.lemma_+")" if token.text != token.lemma else ""):<20}   {token.pos_:>5}    {token.dep_ + "<" + token.head.text + ">":<20}       {token.tag_:<4}   {token.morph}'
    elif offsets:
        s = f'{token.idx:<4} {token.idx + len(token.text):<4} {token.text}'
    else:
        s = token.text
    if index:
        s = index + '    ' + s
    s = s.replace('\n', '\\n')
    return s


def sentence_to_str(sentence, offsets=False, index=None):
    if offsets:
        s = f'{sentence.start:<4} {sentence.end:<4} {sentence.text}'
    else:
        s = sentence.text
    if index:
        s = index + '    ' + s
    s = s.replace('\n', '\\n')
    return s


class PrintsToStderr:
    """
    From https://stackoverflow.com/a/45669280, because trankit prints its debug info...
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = sys.stderr

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


@functools.cache
def load_model(lang: str, trf: bool):

    if trf:
        import spacy_trankit
        with PrintsToStderr():
            model = spacy_trankit.load(lang)
    else:
        try:
            model_name = models[lang]
        except KeyError:
            raise NotImplementedError(f'Language {lang} not supported.')

        try:
            model = spacy.load(model_name)
        except OSError:
            spacy.cli.download(model_name)
            model = spacy.load(model_name)

    return model

