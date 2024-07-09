#!/usr/bin/python

import argparse
import sys
import functools
import logging
import json
import os

import spacy

if '--trf' in sys.argv:
    import spacy_trankit
    from trankit import Pipeline
    from trankit.utils import code2lang, lang2treebank
    from spacy_trankit.tokenizer import TrankitTokenizer
else:
    import spacy_fastlang  # registers the language_detector
    import fasttext
    fasttext.FastText.eprint = lambda x: None  # monkey patch as per https://github.com/facebookresearch/fastText/issues/1067


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

spacy_models = {
    'en': 'en_core_web_sm',
    'nl': 'nl_core_news_sm',
}
trankit_languages = list(spacy_models.keys())


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
        spacy.displacy.serve(docs_for_displacy, style="dep")


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
        spacy.displacy.serve(docs_for_displacy, style="dep")


def spacyjson():
    """
    Process with spaCy, printing (each) resulting doc as json.
    """
    parser = make_base_arg_parser()
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
        spacy.displacy.serve(docs_for_displacy, style="dep")


def text_reader(source, linewise: bool):
    if isinstance(source, str):
        texts = source.splitlines() if linewise else [source]
    else:   # it's stdin
        texts = source if linewise else [source.read()]
    for text in texts:
        yield text.rstrip()


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


def load_spacy_model(lang: str):

    if not lang:
        lang_detector = spacy.blank('en')
        lang_detector.add_pipe("language_detector")

        def automodel(text):
            doc = lang_detector(text)
            detected_lang = doc._.language
            nlp = load_spacy_specific_model(detected_lang)
            return nlp(text)

        return automodel

    else:
        return load_spacy_specific_model(lang)


@functools.cache
def load_spacy_specific_model(lang):

    try:
        model_name = spacy_models[lang]
    except KeyError:
        raise NotImplementedError(f'Language {lang} not supported.')

    try:
        model = spacy.load(model_name)
    except OSError:
        spacy.cli.download(model_name)
        model = spacy.load(model_name)

    return model


def load_trankit_model(lang: str):

    class PrintsToStderr: # From https://stackoverflow.com/a/45669280, because trankit prints its debug info...
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = sys.stderr
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout = self._original_stdout

    with PrintsToStderr():
        if lang:
            model = spacy_trankit.load(lang)
        else:
            model = spacy_trankit.load(trankit_languages[0], **{'@tokenizers': 'spacy_cli.AutoPipelineAsTokenizer.v1'})

    return model


@spacy.util.registry.tokenizers("spacy_cli.AutoPipelineAsTokenizer.v1")
def create_tokenizer(lang: str, cache_dir = None):
    """
    Adapted from spacy-trankit to enable automatic language detect.
    """

    def tokenizer_factory(nlp, lang=lang, cache_dir=cache_dir, **kwargs) -> "TrankitTokenizer":
        load_from_path = cache_dir is not None
        if lang not in lang2treebank and lang in code2lang:
            lang = code2lang[lang]

        if load_from_path:
            if not os.path.exists(cache_dir):
                raise ValueError(
                    f"Path {cache_dir} does not exist. "
                    f"Please download the model and save it to this path."
                )
            model = Pipeline(lang=lang, cache_dir=cache_dir, **kwargs)
        else:
            model = Pipeline(lang=lang, **kwargs)

        # New addition compared to the official package:
        for lang in trankit_languages[1:]:
            model.add(code2lang[lang])
        model.set_auto(True)

        return TrankitTokenizer(model=model, vocab=nlp.vocab)

    return tokenizer_factory

