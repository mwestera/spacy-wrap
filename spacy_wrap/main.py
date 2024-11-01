import functools
import logging
import os
import sys
from typing import Callable, Union


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
    'nl': 'nl_core_news_md',
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


def sentencize(text, language=None, use_trf=False, return_spacy=False, strict_punct='.!?'):
    if isinstance(text, str):
        doc = parse(text, language, use_trf)
    else:   # assume its doc
        doc = text
    sents_to_merge = []
    for sent in doc.sents:
        sents_to_merge.append(sent)
        if not strict_punct or sent.text.strip()[-1] in strict_punct:
            sent = doc[sents_to_merge[0].start:sents_to_merge[-1].end]
            sents_to_merge = []
            yield sent if return_spacy else sent.text


def sentencize_contextual(*args, return_spacy=False, min_n_sent=None, min_n_tokens=None, max_n_tokens=None, block_context: Callable = None, **kwargs):

    min_n_sent = min_n_sent or 0
    min_n_tokens = min_n_tokens or 0
    max_n_tokens = max_n_tokens or 9999999

    sentences = list(sentencize(*args, **kwargs, return_spacy=True))

    for i, sentence in enumerate(sentences):
        sentences_to_use = [sentence]
        n_tokens = len(sentence)

        for previous_sentence in reversed(sentences[:i]):
            if ((block_context and block_context(previous_sentence))
                    or (len(sentences_to_use) >= min_n_sent and n_tokens >= min_n_tokens)
                    or (n_tokens + len(previous_sentence) > max_n_tokens)):
                break
            sentences_to_use.insert(0, previous_sentence)
            n_tokens += len(previous_sentence)

        chunk = sentences_to_use[0].doc[sentences_to_use[0].start:sentences_to_use[-1].end]
        yield chunk if return_spacy else chunk.text


def sentencize_chunked(*args, return_spacy=False, min_n_sent=None, min_n_tokens=None, max_n_tokens=None, **kwargs):

    min_n_sent = min_n_sent or 0
    min_n_tokens = min_n_tokens or 0
    max_n_tokens = max_n_tokens or 9999999

    current_chunk = []
    current_n_tokens = 0

    for sentence in sentencize(*args, **kwargs, return_spacy=True):

        if current_chunk and ((len(current_chunk) >= min_n_sent and current_n_tokens >= min_n_tokens) or current_n_tokens + len(sentence) > max_n_tokens):
            span = current_chunk[0].doc[current_chunk[0].start:current_chunk[-1].end]
            yield span if return_spacy else span.text

            current_chunk = [sentence]
            current_n_tokens = len(sentence)

        current_chunk.append(sentence)
        current_n_tokens += len(sentence)

    if current_chunk:
        span = current_chunk[0].doc[current_chunk[0].start:current_chunk[-1].end]
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

        def pipe(texts):
            for text in texts:
                yield automodel(text)

        automodel.pipe = pipe   # TODO: workaround; need to make it an actual model instance

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
    retur