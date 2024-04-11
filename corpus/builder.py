from enum import Enum

from .type import Type
from .handler import Handler


class Builder:
    """
        A class that builds files from given corpora.

        Attributes:
            _filehandler (Handler): A handler object that handles file operations.
            _corpus (dict): A dictionary containing the corpora.
            _sent_div (set): A set containing the sentence delimiters.
            _tok_delim (str): A string containing the token delimiter.
            _sent_delim (str): A string containing the sentence delimiter.
            _punc_corrections (dict): A dictionary containing the punctuation corrections.
        """

    def __init__(self, tok_delim="\b", sent_delim="\n"):
        self._filehandler = Handler()
        self._corpus = {}
        self._corpus['bijankhan'] = self._filehandler.get_file("bijankhan_unprocessed")
        self._corpus['peykareh'] = self._filehandler.get_file("peykareh_unprocessed")
        self._sent_div = {".", "?", "؟", "!"}
        self._tok_delim = tok_delim
        self._sent_delim = sent_delim
        self._punc_corrections = {' ‹ ': '‹',
                                  ' › ': '›',
                                  ' « ': ' «',
                                  ' » ': '» ',
                                  ' { ': ' {',
                                  ' } ': '} ',
                                  ' [ ': ' [',
                                  ' ] ': '] ',
                                  ' ) ': ' (',
                                  ' ( ': ') ',

                                  ' ؟ ': '؟ ',
                                  ' ؟': "؟ ",
                                  ' ! ': '! ',
                                  ' !': "! ",
                                  ' . ': '. ',
                                  ' .': ". ",
                                  ' ، ': '، ',
                                  ' : ': ': ',
                                  ' ؛ ': '؛',
                                  ' ’ ': '’',
                                  " ' ": "'",
                                  ' … ': '…',
                                  ' & ': '&',
                                  ' - ': '-',
                                  ' \\ ': '\\',
                                  ' * ': ' * ',
                                  ' ^ ': '^',
                                  ' / ': '/',
                                  ' ~ ': '~',
                                  ' " ': '"',
                                  ' $ ': ' $',
                                  ' % ': '%',
                                  ' – ': ' – ',
                                  ' + ': '+',
                                  ' @ ': '@'}

    def __bij_generate_sentence_tokenized(self, tokens, pos):
        """
        generates a list of sentences that each one is divided with at least one of sent_divs

        Howtouse  sentences = list(generator.generate_sentences(lines)
        :param tokens: a list of tokens that
        :return:
        """
        sentence = []
        sent_pos = []
        for i, token in enumerate(tokens):
            if token not in self._sent_div:
                sentence.append(token)
                sent_pos.append(pos[i])
            else:
                # If the token is a divider, add it to the sentence
                sentence.append(token)
                sent_pos.append(pos[i])
                # If the sentence is non-empty, yield it
                try:
                    if tokens[i + 1] in self._sent_div:
                        continue

                except IndexError:
                    pass
                if sentence:
                    yield [sentence, sent_pos]
                    sentence = []
                    sent_pos = []
        if sentence:
            yield sentence, sent_pos

    def _correct_punctuation(self, text):
        """
               Corrects the punctuation in the given text.

               Args:
                   text (str): The text to be corrected.

               Returns:
                   str: The corrected text.
               """
        for incorrect, correct in self._punc_corrections.items():
            text = text.replace(incorrect, correct)
        return text

    def _process_corpus(self, corpus_name):
        """
                Processes the given corpus.

                Args:
                    corpus_name (str): The name of the corpus to be processed.

                Returns:
                    list: A list of sentence tokens.
                """
        with open(self._corpus[corpus_name], "r") as f:
            text = f.read()
        # --------------------------------------- BIJANKHAN ---------------------------------------
        if corpus_name == "bijankhan":
            lines = text.split('\n')
            tokens = ['\u200c'.join(line.split()[:-1]) for line in lines]
            pos = [line.split()[-1:] for line in lines]
            sent_results = list(self.__bij_generate_sentence_tokenized(tokens, pos))

            return sent_results
        # --------------------------------------- PEYKAREH ---------------------------------------
        elif corpus_name == "peykareh":
            # Splitting the text into lines using the newline character as a delimiter and splitting each line into
            # tokens using whitespace as a delimiter
            # Initialize empty lists for tokens and POS
            tokens = []
            pos = []

            # Iterate over each line and split it into tokens and POS
            for line in text.split("\n"):
                parts = line.split()
                if parts:
                    tokens.append(parts[0])
                    pos.append(parts[1])
                else:
                    tokens.append([])
                    pos.append([])

            # Initializing variables
            # overall results
            sent_toks = []
            sent_pos = []
            # initiated for each sentence
            sentence_tokens = []
            sentence_pos = []

            # Iterating over the tokens
            for tok, p in zip(tokens, pos):
                if tok:  # if the list is not empty
                    sentence_tokens.append(tok)
                    sentence_pos.append(p)
                else:  # if the list is empty, indicating end of a sentence
                    sent_toks.append(sentence_tokens)
                    sent_pos.append(sentence_pos)
                    sentence_tokens = []
                    sentence_pos = []

            # add the last sentence if it doesn't end with an empty list
            # if sentence_tokens:
            #     sent_toks.append(sentence_tokens)
            return zip(sent_toks, sent_pos)
        else:
            raise Exception(f"Invalid corpus name: {corpus_name}.")

    def template(self, corpus_name, corpus_type: Enum):
        if corpus_name == "bijankhan":
            if corpus_type.value == Type.whole_raw.value:
                pass
            elif corpus_type.value == Type.whole_tok.value:
                pass
            elif corpus_type.value == Type.sents_raw.value:
                pass
            elif corpus_type.value == Type.sents_tok.value:
                pass

            # --------------------------------------- PEYKAREH ---------------------------------------
        elif corpus_name == "peykareh":
            if corpus_type.value == Type.whole_raw.value:
                pass
            elif corpus_type.value == Type.whole_tok.value:
                pass
            elif corpus_type.value == Type.sents_raw.value:
                pass
            elif corpus_type.value == Type.sents_tok.value:
                pass

    def _save_corpus(self, corpus, corpus_name, corpus_type: Enum):
        # --------------------------------------- BIJANKHAN ---------------------------------------
        file_key = Handler.get_file_key(corpus_name, corpus_type)

        if corpus_type.value == Type.whole_raw.value:
            with open(self._filehandler.get_file(file_key), 'w') as f:
                f.write(corpus)
        elif corpus_type.value == Type.whole_tok.value:
            whole_str = ""
            whole_str += self._tok_delim.join(corpus)
            with open(self._filehandler.get_file(file_key), 'w') as f:
                f.write(whole_str)
        elif corpus_type.value == Type.sents_raw.value:
            with open(self._filehandler.get_file(file_key), 'w') as f:
                for sent in corpus:
                    f.write(sent + self._sent_delim)
        elif corpus_type.value == Type.sents_tok.value:
            with open(self._filehandler.get_file(file_key), 'w') as f:
                # Looping over the sentences
                for i, sent_tok in enumerate(corpus):
                    sent_str = self._tok_delim.join(sent_tok)
                    f.write(sent_str + self._sent_delim)

    def build_corpus(self, corpus_name, corpus_type: Enum):
        """
                Builds the corpus based on the given corpus name and type.

                Args:
                    corpus_name (str): The name of the corpus to be built.
                    corpus_type (Enum): The type of the corpus to be built.

                Returns:
                    list
        """
        if corpus_type not in list(Type):
            raise Exception("invalid corpus type requested")
        sent_results= self._process_corpus(corpus_name)
        sent_toks=[]
        sent_pos=[]
        for res in sent_results:
            sent_toks.append(res[0])
            sent_pos.append(res[1])
        del sent_results
        if corpus_type.value == Type.whole_raw.value:
            toks = [t for sentence in sent_toks for t in sentence]
            whole = " ".join(toks)
            whole = self._correct_punctuation(whole)
            self._save_corpus(whole, corpus_name, corpus_type)
            return whole

        elif corpus_type.value == Type.whole_tok.value:
            tokens = [t for sentence in sent_toks for t in sentence]
            self._save_corpus(tokens, corpus_name, corpus_type)
            return tokens

        elif corpus_type.value == Type.sents_raw.value:
            sentences = [" ".join(sentence) for sentence in sent_toks]
            sentences = [self._correct_punctuation(sentence) for sentence in sentences]
            self._save_corpus(sentences, corpus_name, corpus_type)
            return sentences

        elif corpus_type.value == Type.sents_tok.value:
            # self._save_corpus(sent_toks, corpus_name, corpus_type)
            return sent_toks, sent_pos

    def build_all(self):
        corpus_names = self._filehandler.corpus_names()
        corpus_types = self._filehandler.corpus_types()

        combinations = [(corpus_names[i], corpus_types[j]) for i in range(len(corpus_names)) for j in
                        range(len(corpus_types))]
        for (n, t) in combinations:
            self.build_corpus(n, t)
