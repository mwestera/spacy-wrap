# spacy-wrap

Convenient command-line interface for applying spacy and trankit pipelines. Allows word tokenization and sentence segmentation, for given language or using automatic language detection, as well as printing spacy parses directly as json.

## Install

```bash
$ pipx install git+https://github.com/mwestera/spacy-wrap
```

This will make three commands available:

- `tokenize`
- `sentencize`
- `spacyjson`

## Examples

```bash
$ echo "Here's just a short text. For you to parse." | tokenize --info -tree
```

Or, to process each line from a file separately (and this time using a transformer model, --trf):

```bash
$ cat some_dutch_sentences.txt | tokenize --info --trf --lang nl --lines --tree
```

```bash
$ cat texts_in_various_languages.txt | sentencize --trf --lines
```

Note: In this case, will detect language separately for each input line.

Or output full sentence parses in json format:

```bash
$ cat texts_in_various_languages.txt | sentencize --lines --lang nl --json
```

Or entire spacy docs as json:

```bash
$ cat texts_in_various_languages.txt | spacyjson --lines --lang nl --json > parses.jsonl
```
