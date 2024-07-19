import functools
import logging
import os
import sys


import spacy
import spacy_trankit
from spacy_trankit.tokenizer import TrankitTokenizer
import spacy_fastlang  # registers the language_detector

import fasttext
fasttext.FastText.eprint = lambda x: None  # monkey patch as per https://github.com/facebookresearch/fastText/issues/1067

from trankit import Pipeline
from trankit.utils import code2lang, lang2treebank


from spacy import displacy

import tempfile
import webbrowser


# TODO: Where do cached models go?
# TODO: option to set language globally



spacy_models = {
    'en': 'en_core_web_sm',
    'nl': 'nl_core_news_sm',
}
trankit_languages = list(spacy_models.keys())



def parse(text, language=None, use_trf=False):
    nlp = load_trankit_model(language) if use_trf else load_spacy_model(language)
    doc = nlp(text)
    return doc


def tokenize(text, language=None, use_trf=False, return_spacy=False):
    doc = parse(text, language, use_trf)
    for tok in doc:
        yield tok if return_spacy else tok.text


def sentencize(text, language=None, use_trf=False, return_spacy=False, include_previous=0):
    doc = parse(text, language, use_trf)
    sentences = list(doc.sents)

    n_previous = min(include_previous, len(sentences))
    padded_sentences = [None for _ in range(n_previous)] + sentences
    sentence_ngrams = [tuple(filter(None, ngram)) for ngram in zip(*[padded_sentences[i:] for i in range(n_previous + 1)])]

    for ngram in sentence_ngrams:
        span = doc[ngram[0].start:ngram[-1].end]
        yield span if return_spacy else span.text


@functools.cache
def load_spacy_model(lang: str = None):

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
def load_spacy_specific_model(lang: str):

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


@functools.cache
def load_trankit_model(lang: str = None):

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


def render_parse_tree_html(docs):
    html = displacy.render(docs, style="dep")
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        url = 'file://' + f.name
        f.write(html)
    webbrowser.open(url)


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

