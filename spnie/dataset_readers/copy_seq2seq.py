import logging
from typing import List, Dict

import numpy as np
from overrides import overrides
from spnie import bert_utils 

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("copy_seq2seq")
class CopySeq2SeqNetDatasetReader(DatasetReader):
    def __init__(self,
                 target_namespace: str,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 max_tokens: int = None,
                 bert: bool = False,
                 max_extractions: int = 10,
                 validation: bool = False,
                 gradients: bool = True) -> None:
        super().__init__(lazy)
        self._target_namespace = target_namespace
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._max_extractions = max_extractions
        self._max_tokens = max_tokens
        self._bert = bert
        self._validation = validation
        self._gradients = gradients
        global START_SYMBOL, END_SYMBOL
        START_SYMBOL, END_SYMBOL = bert_utils.init_globals()
        if self._bert:
            self._target_token_indexers: Dict[str, TokenIndexer] = source_token_indexers
        else:
            self._target_token_indexers: Dict[str, TokenIndexer] = {
                "tokens": SingleIdTokenIndexer(namespace=self._target_namespace)
            }

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            target_sequences, confidences = [], []
            lines = data_file.readlines() + ['']
            old_sentence = sentence = ''

            for line_num, line in enumerate(lines):
                line = line.strip("\n")
                if line_num != len(lines) - 1:
                    if self._validation:
                        sentence = line
                        extraction = 'dummy'
                        confidence = 1
                    else:
                        sentence, extraction, confidence = line.split('\t')
                        confidence = float(confidence)
                else:
                    sentence, extraction, confidence = '', '', 1

                if line_num == 0:
                    old_sentence = sentence

                if old_sentence != sentence:
                    source_sequence = old_sentence
                    if self._validation:
                        target_sequences = [0] #*self._max_extractions

                    if len(target_sequences) <= self._max_extractions:
                        while len(target_sequences) < self._max_extractions:
                            target_sequences.append('')
                            confidences.append(float(1))
                        if self._validation:
                            target_sequences = None 
                        instance = self.text_to_instance(source_sequence, target_sequences, line_num - 1, \
                                        validation=self._validation, gradients=self._gradients, confidences=confidences)
                        if instance != None:
                            yield instance
                    old_sentence = sentence
                    target_sequences = []
                    confidences = []

                target_sequences.append(extraction)
                confidences.append(float(confidence))


    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text, len(ids)))
        return out

    @overrides
    def text_to_instance(self, source_string: str, target_strings: str = None, example_id: str = None,
                         validation: bool = False, gradients: bool = False,
                         confidences: float = None) -> Instance:
        # pylint: disable=arguments-differ
        if self._bert:
            source_string = bert_utils.replace_strings(source_string)
            if target_strings is not None:
                rep_target_strings = []
                for target_string in target_strings:
                    rep_target_strings.append(bert_utils.replace_strings(target_string))
                target_strings = rep_target_strings

        tokenized_source = self._source_tokenizer.tokenize(source_string)
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        
        source_field = TextField(tokenized_source, self._source_token_indexers)

        if(self._max_tokens != None and len(tokenized_source) > self._max_tokens):
            return None

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(tokenized_source[1:-1], self._target_namespace)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source[1:-1]], "example_ids": example_id,
                       "validation": validation, "gradients": gradients, "confidences": confidences}
        fields_dict = {
            "source_tokens": source_field,
            "source_to_target": source_to_target_field,
        }

        if target_strings is not None:
            target_fields, tokenized_targets, target_token_idss = [], [], []
            for i in range(len(target_strings)):
                tokenized_target = self._target_tokenizer.tokenize(target_strings[i])
                tokenized_target.insert(0, Token(START_SYMBOL))
                tokenized_target.append(Token(END_SYMBOL))
                tokenized_targets.append(tokenized_target)
                target_field = TextField(tokenized_target, self._target_token_indexers)
                target_fields.append(target_field)

                source_and_target_token_ids = self._tokens_to_ids(tokenized_source[1:-1] +
                                                                  tokenized_target)
                source_token_ids = source_and_target_token_ids[:len(tokenized_source) - 2]

                target_token_ids = source_and_target_token_ids[len(tokenized_source) - 2:]
                target_token_idss.append(ArrayField(np.array(target_token_ids)))

            fields_dict["target_tokens"] = ListField(target_fields)
            meta_fields["target_tokens"] = [[y.text for y in tokenized_target[1:-1]] for tokenized_target in tokenized_targets]
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))
            fields_dict["target_token_ids"] = ListField(target_token_idss)
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source[1:-1])
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))

        fields_dict["metadata"] = MetadataField(meta_fields)
        return Instance(fields_dict)