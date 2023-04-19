import logging
from typing import Dict, Tuple, List, Any, Union
import argparse
import numpy as np
import time
from overrides import overrides
import ipdb
import pdb
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell, GRUCell
from torch.nn import LSTM
from scipy.optimize import linear_sum_assignment

import allennlp
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, MatrixAttention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric, BLEU
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.div_beam_search import DivBeamSearch
from allennlp.nn.cov_beam_search import CoverageBeamSearch

from transformers.modeling_bert import BertConfig, BertIntermediate, BertOutput, BertAttention, BertLayerNorm, BertSelfAttention


from spnie import bert_utils


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class KeyDict(dict):
    def __missing__(self, key):
        return key


@Model.register("copy_seq2seq_spnie")
class CopyNetSeq2Seq(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 beam_size: int,
                 max_decoding_steps: int,
                 loss_func: str,
                 set_attention_mode: str,
                 num_generated_triples: int = 10,
                 set_decoder_layers: int = 3,
                 target_embedding_dim: int = 100,
                 decoder_layers: int = 3,
                 copy_token: str = "@COPY@",
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 tensor_based_metric: Metric = None,
                 token_based_metric: Metric = None,
                 lambda_diversity: int = 5,
                 beam_search_type: str = "beam_search",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 bert: bool = False,
                 decoder_config: str = '',
                 decoder_type: str = 'lstm',
                 teacher_forcing: bool = True):
        super().__init__(vocab)
        self._decoder_type = decoder_type
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._bert = bert
        global START_SYMBOL, END_SYMBOL
        
        if self._bert:
            START_SYMBOL, END_SYMBOL = bert_utils.init_globals()
            self._target_vocab_size = 28996
            self.token_mapping = bert_utils.mapping
            self.config = BertConfig.from_json_file('./bert-base-cased/config.json')
            # self.config.hidden_dropout_prob = 0.0
            # self.config.attention_probs_dropout_prob = 0.0

        else:
            if self.vocab.get_token_index(copy_token, self._target_namespace) == 1:
                self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace) + 1
            else:
                self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)
            self.token_mapping = KeyDict()

        # Encoding modules.
        
        self._beam_size = beam_size
        self._max_decoding_steps = max_decoding_steps
        self._loss_func = loss_func
        self._set_attention_mode = set_attention_mode
        self._target_embedding_dim = target_embedding_dim
        self._decoder_layers = decoder_layers
        self._copy_token = copy_token
        self._tensor_based_metric = tensor_based_metric
        self._token_based_metric = token_based_metric
        self._lambda_diversity = lambda_diversity
        self._beam_search_type = beam_search_type
        self._initializer = initializer
        self._decoder_config = decoder_config
        self._decoder_type = decoder_type
        self._teacher_forcing = teacher_forcing
        self.ngt = num_generated_triples
        
        self._source_embedder = source_embedder
        self._encoder = encoder
        self.hidden_size = self._encoder.get_output_dim()
        self.config.hidden_size = self.hidden_size

        self._set_decoder_layers = set_decoder_layers
        self.config.num_attention_heads = 4
        self._set_decoder = SetDecoder(self.config, self.ngt, self._set_decoder_layers)
        
        # triple decoder layer
        self._target_embedder = Embedding(self._target_vocab_size, self._target_embedding_dim)
        self.LayerNorm = BertLayerNorm(self._target_embedding_dim, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        
        self.config.output_attentions = True
        self.config.num_attention_heads = 1
        self._attention = BertAttention(self.config)
        self._input_projection_layer = Linear(self._target_embedding_dim + self.hidden_size * 3, self.hidden_size)

        if self._decoder_type == 'lstm':
            self._decoder_cell = LSTM(self.hidden_size, self.hidden_size, num_layers=self._decoder_layers, batch_first=True)
            self._decoder_dropout = nn.Dropout(self.config.hidden_dropout_prob)
        elif self._decoder_type == 'transformer':
            decoder_layer = torch.nn.TransformerDecoderLayer(d_model=256, nhead=4)
            self._decoder_cell = torch.nn.TransformerDecoder(decoder_layer, num_layers=1)
            self._decoder_dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self._output_generation_layer = Linear(self.hidden_size, self._target_vocab_size)
        self._output_copying_layer = Linear(self.hidden_size, self.hidden_size)

        #set prediction ---- attention
        if self._set_attention_mode == 'dynamic-attention':
            self._set_attention = nn.Sequential(
                nn.Linear(self.ngt, 3),
                nn.Linear(3, self.ngt)
                )
        elif self._set_attention_mode == 'self-attention':
            self.config.output_attentions = True
            self.config.num_attention_heads = 1
            self.config.hidden_size = self._target_embedding_dim
            self._set_attention = BertAttention(self.config)

        self._query_output_layer = nn.Linear(self.config.hidden_size, 2)

        self.log_var_a = torch.zeros((1,), requires_grad=True)
        self.log_var_b = torch.zeros((1,), requires_grad=True)

        self._initializer(self)
        self._initialized = False
    
    def _leave_one(self, tokens, index):
        # Remove all 'index' tokens except one
        # In-order to allow extraction of separate triples
        # does not operate on batches
        unpadded_tokens = []
        prev_token = -1
        for token in tokens:
            if prev_token == index and token == index:
                continue
            unpadded_tokens.append(token)
            prev_token = token
        return unpadded_tokens


    def initialize(self):
        self._src_start_index = self.vocab.get_token_index(START_SYMBOL, self._source_namespace)
        self._src_end_index = self.vocab.get_token_index(END_SYMBOL, self._source_namespace)
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        if self._bert:
            self.vocab._oov_token = '[unused99]'
            self.vocab._padding_token = '[PAD]'
        self._oov_index = self.vocab.get_token_index(self.vocab._oov_token,
                                                     self._target_namespace)  # pylint: disable=protected-access
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token,
                                                     self._target_namespace)  # pylint: disable=protected-access

        self._copy_index = self.vocab.add_token_to_namespace(self.token_mapping[self._copy_token],
                                                             self._target_namespace)
        self._eoe_index = self.vocab.get_token_index(self.token_mapping['EOE'], self._target_namespace)

        self.start_arg1 = self.vocab.get_token_index(self.token_mapping['<arg1>'], self._target_namespace)
        self.end_arg1 = self.vocab.get_token_index(self.token_mapping['</arg1>'], self._target_namespace)
        self.start_arg2 = self.vocab.get_token_index(self.token_mapping['<arg2>'], self._target_namespace)
        self.end_arg2 = self.vocab.get_token_index(self.token_mapping['</arg2>'], self._target_namespace)
        self.start_rel = self.vocab.get_token_index(self.token_mapping['<rel>'], self._target_namespace)
        self.end_rel = self.vocab.get_token_index(self.token_mapping['</rel>'], self._target_namespace)

        if self._beam_search_type == 'beam_search':
            self._beam_search = BeamSearch(self._end_index, max_steps=self._max_decoding_steps,
                                           beam_size=self._beam_size)
        elif self._beam_search_type == 'div_beam_search':
            self._beam_search = DivBeamSearch(self._end_index, max_steps=self._max_decoding_steps,
                                              beam_size=self._beam_size, lambda_diversity=self._lambda_diversity, \
                                              ignore_indices=[self.start_arg1, self.start_arg2, self.start_rel,
                                                              self.end_arg1, self.end_arg2, self.end_rel])
        elif self._beam_search_type == 'cov_beam_search':
            self._beam_search = CoverageBeamSearch(self._end_index, max_steps=self._max_decoding_steps,
                                                   beam_size=self._beam_size)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                source_token_ids: torch.Tensor,
                source_to_target: torch.Tensor,
                metadata: List[Dict[str, Any]],
                target_tokens: Dict[str, torch.LongTensor] = None,
                target_token_ids: torch.Tensor = None,
                optimizer=None) -> Dict[str, torch.Tensor]:
        if not self._initialized:
            self.initialize()
            self._initialized = True
        if self.training and not self._decoder_type == 'transformer':
            self._decoder_cell.flatten_parameters()
        
        state = self._encode(source_tokens)
        state["source_to_target"] = source_to_target
        state, set_pred_mask = self._set_decoder(state)    # source_mask, encoder_outputs, set_hs, set_pred_mask
        # print(source_to_target.size(), target_tokens['tokens'].size())
        # print(state['set_hs'].sum(-1))
        # print(set_pred_mask.max(-1)[-1])

        if target_tokens:
            target_mask = (target_tokens['tokens'] > 0).float()
            target_num_mask = (target_mask.sum(-1) > 0).long()
            max_decoding_steps = int(target_mask.sum(-1).max() - 1)
            
            loss_matrix, similarity_matrix = self._tri_decoder(state, target_tokens["tokens"], target_mask, max_decoding_steps)
            output_dict = self._criterion(set_pred_mask, loss_matrix, similarity_matrix, target_mask, target_num_mask, max_decoding_steps, state)
            
        else:
            output_dict = {}
            predictions = self._forward_beam_search(state, set_pred_mask)
            output_dict.update(predictions)
        
        output_dict["metadata"] = metadata
        if metadata[0]['validation']:
            predicted_tokens = self._get_predicted_tokens(output_dict["predictions"],
                                                          metadata,
                                                          n_best=5)
            # for i in range(len(predicted_tokens)):
            #     print(predicted_tokens[i])
            # aaa
            predicted_confidences = output_dict['predicted_log_probs']
            self._token_based_metric(predicted_tokens, predicted_confidences,  # type: ignore
                                     [x["example_ids"] for x in metadata])
        return output_dict


    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embedded_input = self._source_embedder(source_tokens) # (bz, msl, hs)
        source_mask = util.get_text_field_mask(source_tokens)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_hs": encoder_outputs}


    def _tri_decoder(self, state, target_tokens, target_mask, max_decoding_steps):
        bz, msl, hs = state["encoder_hs"].size()
        cur_ngt, mtl =target_tokens.size()[1:]
        ngt = self.ngt
        nt = self._decoder_layers

        state["set_hs"] = state["set_hs"].unsqueeze(2).repeat(1, 1, cur_ngt, 1).view(bz * ngt * cur_ngt, hs) # [bz * ngt * cur_ngt, hs]
        #state["decoder_hidden"] = state["set_hs"]
        state["decoder_hidden"] = state["encoder_hs"][:, 0, :].unsqueeze(1).repeat(1, ngt * cur_ngt, 1).view(bz * ngt * cur_ngt, hs) # [bz * ngt * cur_ngt, hs]
        state["decoder_hidden_all"] = state["decoder_hidden"].unsqueeze(0).repeat(nt, 1, 1)   # [nt, bz * ngt * cur_ngt, hs]
        state["decoder_context"] = state["encoder_hs"].new_zeros(bz * ngt * cur_ngt, hs) # [bz * ngt * cur_ngt, hs]
        state["decoder_context_all"] = state["encoder_hs"].new_zeros(nt, bz * ngt * cur_ngt, hs) # [nt, bz * ngt * cur_ngt, hs]
        state["copy_log_probs"] = (state["encoder_hs"].new_zeros(bz * ngt * cur_ngt, msl - 2) + 1e-45).log()

        generation_mask = state["encoder_hs"].new_full((bz * ngt * cur_ngt, self._target_vocab_size), fill_value = 1.0) # [bz * ngt * cur_ngt, vocab_size] 
        copy_mask = state["source_mask"][:, 1:-1].float().unsqueeze(1).repeat(1, ngt * cur_ngt, 1).view(bz * ngt * cur_ngt, -1) # [bz * ngt * cur_ngt, msl-2]
        selective_weights = state["encoder_hs"].new_zeros(copy_mask.size()) # [bz * ngt * cur_ngt, msl-2]

        state["source_mask"] = state["source_mask"].unsqueeze(1).repeat(1, ngt * cur_ngt, 1).view(bz * ngt * cur_ngt, msl)
        target_tokens = target_tokens.unsqueeze(1).repeat(1, ngt, 1, 1).view(bz * ngt * cur_ngt, mtl)
        target_mask = target_mask.unsqueeze(1).repeat(1, ngt, 1, 1).view(bz * ngt * cur_ngt, mtl)
        state["encoder_hs"] = state["encoder_hs"].unsqueeze(1).repeat(1, ngt * cur_ngt, 1, 1).view(bz * ngt * cur_ngt, -1, hs) # [bz * ngt * cur_ngt, msl, hs]
        state["source_to_target"] = state["source_to_target"].unsqueeze(1).repeat(1, ngt * cur_ngt, 1).view(bz * ngt * cur_ngt, msl-2)

        step_log_likelihoods = state["encoder_hs"].new_zeros(bz * ngt * cur_ngt, max_decoding_steps)
        
        for t in range(max_decoding_steps):
            dec_inp = self._target_embedder(target_tokens[:, t])
            dec_inp = self.dropout(self.LayerNorm(dec_inp))
            step_target_tokens = target_tokens[:, t + 1]
            
            if t < max_decoding_steps - 1:
                target_to_source = state["source_to_target"] == step_target_tokens.unsqueeze(-1)
            
            log_probs, state = self._decoder(dec_inp, selective_weights, generation_mask, copy_mask, state)
            generation_log_probs, copy_log_probs = log_probs.split([self._target_vocab_size, msl - 2], dim=-1) # [bz * ngt, vocab_size] [bz * ngt, msl - 2]
            state["copy_log_probs"] = copy_log_probs
            selective_weights = util.masked_softmax(log_probs[:, self._target_vocab_size:], target_to_source)

            core_copy_log_probs = copy_log_probs + (target_to_source.float() + 1e-45).log()
            core_generation_log_probs = log_probs.gather(1, step_target_tokens.unsqueeze(1))
            core_combined_gen_and_copy = torch.cat((core_generation_log_probs, core_copy_log_probs), dim=-1)
            log_likelihood = util.logsumexp(core_combined_gen_and_copy)
            step_log_likelihoods[:, t] = log_likelihood

        target_mask = target_mask[:, 1: max_decoding_steps + 1].float()
        log_likelihood = (step_log_likelihoods * target_mask).sum(dim=-1)
        
        loss_matrix = - log_likelihood
        loss_matrix = loss_matrix.view(bz, ngt, cur_ngt)
        
        similarity_matrix = - log_likelihood/ (target_mask.sum(-1) + 1)
        similarity_matrix = similarity_matrix.view(bz, ngt, cur_ngt)

        # mask = (target_mask.sum(-1)==2.).float().view(bz, ngt, cur_ngt)
        # similarity_matrix = similarity_matrix * mask * 10000 + similarity_matrix * (1-mask)
        # loss_matrix = loss_matrix * mask / (mask.sum(-1).unsqueeze(-1)+1) + loss_matrix * (1-mask)
        # loss_matrix = loss_matrix * mask / 8. + loss_matrix * (1-mask)

        return loss_matrix, similarity_matrix
    

    def _decoder(self, y_prev, selective_weights, generation_mask, copy_mask, state):
        bz, msl, hs = state["encoder_hs"].size()
        ngt = self.ngt
        nt = self._decoder_layers

        encoder_extended_attention_mask = (1.0 - state["source_mask"][:, None, None, :]) * -10000.0
        cross_attention_outputs = self._attention(hidden_states=state["decoder_hidden"].view(bz, 1, hs), 
                encoder_hidden_states=state["encoder_hs"], 
                encoder_attention_mask=encoder_extended_attention_mask)
        attentive_read = cross_attention_outputs[0].view(bz, -1)  # [bz * ngt * cur_ngt, hs]
        #attentive_weights = self._attention(state["decoder_hidden"], state["encoder_hs"], state["source_mask"].float())
        #attentive_read = util.weighted_sum(state["encoder_hs"], attentive_weights)  # [bz * ngt * cur_ngt, hs]

        selective_read = selective_weights.unsqueeze(1).bmm(state["encoder_hs"][:, 1:-1]).squeeze(1)

        decoder_input = torch.cat((y_prev, attentive_read, selective_read, state["set_hs"]), -1)
        #decoder_input = torch.cat((y_prev, attentive_read, selective_read), -1)
        projected_decoder_input = self._input_projection_layer(decoder_input) # [bz * ngt * cur_ngt, hs]
        
        _, (state["decoder_hidden_all"], state["decoder_context_all"]) = self._decoder_cell(projected_decoder_input.unsqueeze(1), (state["decoder_hidden_all"], state["decoder_context_all"]))

        state["decoder_hidden"] = self._decoder_dropout(state["decoder_hidden_all"][-1])
        state["decoder_context"] = state["decoder_context_all"][-1]

        generation_scores = self._output_generation_layer(state["decoder_hidden"]) # [bz * ngt * cur_ngt, vocab_size]
        
        copy_projection = torch.tanh(self._output_copying_layer(state["encoder_hs"][:, 1:-1])) # [bz * ngt * cur_ngt, msl-2, hs]
        copy_scores = copy_projection.bmm(state["decoder_hidden"].unsqueeze(-1)).squeeze(-1) # [bz * ngt * cur_ngt, msl-2]
        

        mask = torch.cat((generation_mask, copy_mask), -1) # [bz * ngt * cur_ngt, vocab_size + msl - 2] 
        all_scores = torch.cat((generation_scores, copy_scores), -1) # [bz * ngt * cur_ngt, vocab_size + msl - 2]
        log_probs = util.masked_log_softmax(all_scores, mask)
        return log_probs, state


    def _criterion(self, set_pred_mask, sim_mat, loss_mat, target_mask, target_num_mask, max_decoding_steps, state):
        # set_pred_mask [bz, ngt, 2]
        # target_num_mask [bz, cur_ngt]
        # dec_embedded [bz, ngt, cur_ngt, self._target_embedding_dim]
        bz, ngt, cur_ngt = sim_mat.size()
        
        for i in range(bz):
            for j in range(ngt):
                for k in range(cur_ngt):
                    if int(target_num_mask[i, k]) == 0:
                        sim_mat[i, j, k] = float('inf')

        sim_mat_cpu = sim_mat.cpu().detach().numpy()
        indices = [linear_sum_assignment(c[:, :int(sum(k))]) for i, (c, k) in enumerate(zip(sim_mat_cpu, target_num_mask))]
        #indices = [linear_sum_assignment(c[:int(sum(k[:])), :int(sum(k[:]))]) for i, (c, k) in enumerate(zip(sim_mat_cpu, target_num_mask))] # only use ordered output
        #print(target_mask.sum(-1))
        target_loss = 0
        probs = []
        for i in range(bz):
            prob = 0
            for j in range(len(indices[i][0])):
                x = indices[i][0][j]
                y = indices[i][1][j]
                if target_mask[i, y].sum()==2.:
                    weight = 0.1
                else:
                    weight = 1.
                
                target_loss += weight * sim_mat[i, x, y]
                prob += -float(sim_mat[i, x, y])
            probs.append(prob)
        target_loss = target_loss/bz
        
        return {"loss": target_loss, "probs": probs}

    # def _criterion(self, set_pred_mask, sim_mat, loss_mat, target_mask, target_num_mask, max_decoding_steps, state):
    #     # set_pred_mask [bz, ngt, 2]
    #     # target_num_mask [bz, cur_ngt]
    #     # dec_embedded [bz, ngt, cur_ngt, self._target_embedding_dim]
    #     bz, ngt, cur_ngt = sim_mat.size()
    #     gold_target_position = target_num_mask.new_zeros((bz, ngt))
        
    #     for i in range(bz):
    #         for j in range(ngt):
    #             for k in range(cur_ngt):
    #                 if int(target_num_mask[i, k]) == 0:
    #                     sim_mat[i, j, k] = float('inf')

    #     sim_mat_cpu = sim_mat.cpu().detach().numpy()
    #     indices = [linear_sum_assignment(c[:, :int(sum(k))]) for i, (c, k) in enumerate(zip(sim_mat_cpu, target_num_mask))]
    #     #indices = [linear_sum_assignment(c[:int(sum(k[:])), :int(sum(k[:]))]) for i, (c, k) in enumerate(zip(sim_mat_cpu, target_num_mask))] # only use ordered output

    #     target_loss = 0
    #     probs = []
    #     count = 0
    #     for i in range(bz):
    #         prob = 0
    #         for j in range(len(indices[i][0])):
    #             x = indices[i][0][j]
    #             y = indices[i][1][j]
    #             target_loss += sim_mat[i, x, y]
    #             prob += -float(sim_mat[i, x, y])
    #             # target_loss += sim_mat[i, j, j]
    #             # prob += -float(sim_mat[i, j, j])
    #             gold_target_position[i, indices[i][0][j]] = 1.
    #             count += 1
    #         probs.append(prob)
    #     target_loss = target_loss/bz

    #     # set loss
    #     #set_loss_weight = torch.Tensor([0.2, 0.8]).to(set_pred_mask.device)
    #     set_loss_weight = self.calculate_loss_weight(gold_target_position)
    #     set_loss = 0
    #     for i in range(bz):
    #         set_loss += F.cross_entropy(set_pred_mask[i], gold_target_position[i], weight=set_loss_weight)
    #     set_loss = set_loss/bz

    #     self.log_var_a = self.log_var_a.to(set_loss.device)
    #     self.log_var_b = self.log_var_b.to(set_loss.device)
    #     total_loss = torch.exp(-self.log_var_a) * set_loss + torch.exp(-self.log_var_b) * target_loss + self.log_var_a + self.log_var_b
        
    #     # print('\n')
    #     # # # print(set_loss_weight)
    #     # print(set_pred_mask.max(-1)[-1])
    #     # print(gold_target_position)
    #     # print(target_loss, set_loss)
        
    #     return {"loss": total_loss, "probs": probs}


    def _forward_beam_search(self, state, set_pred_mask):
        bz, msl, hs = state["encoder_hs"].size()
        ngt = self.ngt
        nt = self._decoder_layers
        beam_size = self._beam_size
        per_node_beam_size = beam_size
        encoder_hs = state["encoder_hs"]

        state["source_to_target"] = state["source_to_target"].unsqueeze(1).repeat(1, ngt, 1).view(bz * ngt, msl-2)
        state["source_mask"] = state["source_mask"].unsqueeze(1).repeat(1, ngt, 1).view(bz * ngt, msl)
        state["encoder_hs"] = encoder_hs.unsqueeze(1).repeat(1, ngt, 1, 1).view(bz * ngt, msl, hs)
         
        state["set_hs"] = state["set_hs"].view(bz * ngt, hs)# [bz * ngt, hs]
        #state["decoder_hidden"] = state["set_hs"]
        state["decoder_hidden"] = encoder_hs[:, 0, :].unsqueeze(1).repeat(1, ngt, 1).view(bz * ngt, hs) # [bz * ngt, hs]
        state["decoder_context"] = encoder_hs.new_zeros(bz * ngt, hs) # [bz * ngt, hs]
        state["decoder_hidden_all"] = state["decoder_hidden"].unsqueeze(0).repeat(nt, 1, 1).transpose(0,1).contiguous().view(bz * ngt, nt * hs)   # [nt, bz * ngt, hs]
        state["decoder_context_all"] = encoder_hs.new_zeros(bz * ngt, nt * hs) # [bz * ngt, nt * hs]

        state["copy_log_probs"] = (encoder_hs.new_zeros(bz * ngt, msl - 2) + 1e-45).log()
        # beam search
        start_predictions = state["source_mask"].new_full((bz * ngt,), fill_value=self._start_index)  # start of triple
        #  [bz * ngt, beam_size, max_decoding_step]  [bz * ngt, beam_size]
        all_top_k_predictions, log_probabilities = self._beam_search.search(start_predictions, state, self.take_search_step)
        #print(log_probabilities[10:20])
        target_mask = all_top_k_predictions != self._end_index
        log_probabilities = log_probabilities / (target_mask.sum(dim=2).float() + 1)
        
        new_predictions = all_top_k_predictions.new_ones(bz * ngt, beam_size, self._max_decoding_steps) * self._end_index
        for i in range(bz*ngt):
            for j in range(beam_size):
                for k in range(all_top_k_predictions.size()[2]):
                    new_predictions[i][j][k] = all_top_k_predictions[i][j][k]
        max_probs, max_indices = log_probabilities.max(-1)
        predictions_log_probs = max_probs.view(bz, -1)
        max_indices = max_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self._max_decoding_steps)
        predictions = new_predictions.gather(1, max_indices).view(bz, ngt, self._max_decoding_steps)

        # predictions = new_predictions[:, 0, :].view(bz, ngt, -1)
        # predictions_log_probs = log_probabilities[:, 0].view(bz, -1)

        # catenatate all the predictions
        # set_pred_mask = set_pred_mask.max(-1)[1]
        # set_pred_mask = set_pred_mask.unsqueeze(-1).repeat(1, 1, predictions.size()[-1])
        # invalid_predictions = predictions.new_ones((bz, ngt, predictions.size()[-1])) * self._end_index
        # predictions = predictions * set_pred_mask + invalid_predictions * (1 - set_pred_mask)
        
        end_column = predictions.new_ones(bz, ngt, 1) * self._end_index
        predictions = torch.cat((predictions, end_column), dim=-1)
        predictions = predictions.view(bz, -1).unsqueeze(dim=1)

        output_dict = {"predicted_log_probs": predictions_log_probs, "predictions": predictions}

        return output_dict


    def take_search_step(self, last_predictions, state):
        bz, msl, hs = state["encoder_hs"].size()
        nt = self._decoder_layers

        group_size, trimmed_source_length = state["source_to_target"].size()
        expanded_last_predictions = last_predictions.unsqueeze(-1).expand(group_size, trimmed_source_length)
        mask = (state["source_to_target"] == expanded_last_predictions).long()
        selective_weights = util.masked_softmax(state["copy_log_probs"], mask)
        state = self._decoder_step(last_predictions, selective_weights, state)
        
        generation_scores = self._output_generation_layer(state["decoder_hidden"]) # [bz * ngt, vocab_size]
        copy_projection = torch.tanh(self._output_copying_layer(state["encoder_hs"][:, 1:-1])) # [bz * ngt, msl-2, hs]
        copy_scores = copy_projection.bmm(state["decoder_hidden"].unsqueeze(-1)).squeeze(-1) # [bz * ngt, msl-2]
        all_scores = torch.cat((generation_scores, copy_scores), -1) # [bz * ngt, vocab_size + msl - 2]

        generation_mask = state["encoder_hs"].new_full((bz, self._target_vocab_size), fill_value = 1.0) # [bz * ngt, vocab_size] 
        copy_mask = state["source_mask"][:, 1:-1].float() # [bz * ngt, msl-2]
        mask = torch.cat((generation_mask, copy_mask), -1) # [bz * ngt, vocab_size + msl - 2] 
        log_probs = util.masked_log_softmax(all_scores, mask)
        generation_log_probs, copy_log_probs = log_probs.split([self._target_vocab_size, trimmed_source_length], dim=-1) # [bz * ngt, vocab_size] [bz * ngt, msl - 2]
        state["copy_log_probs"] = copy_log_probs

        final_log_probs = self._gather_final_log_probs(generation_log_probs, copy_log_probs, state)
        return final_log_probs, state


    def _gather_final_log_probs(self, generation_log_probs, copy_log_probs, state):
        _, trimmed_source_length = state["source_to_target"].size()
        source_token_ids = state["source_to_target"]

        modified_log_probs_list: List[torch.Tensor] = []
        for i in range(trimmed_source_length):
            # shape: (group_size,)
            copy_log_probs_slice = copy_log_probs[:, i]
            source_to_target_slice = state["source_to_target"][:, i]
            copy_log_probs_to_add_mask = (source_to_target_slice != 50000).float()
            copy_log_probs_to_add = copy_log_probs_slice + (copy_log_probs_to_add_mask + 1e-45).log()
            # shape: (batch_size, 1)
            copy_log_probs_to_add = copy_log_probs_to_add.unsqueeze(-1)
            # shape: (batch_size, 1)
            selected_generation_log_probs = generation_log_probs.gather(1, source_to_target_slice.unsqueeze(-1))
            combined_scores = util.logsumexp(
                    torch.cat((selected_generation_log_probs, copy_log_probs_to_add), dim=1))
            generation_log_probs = generation_log_probs.scatter(-1,
                                                                source_to_target_slice.unsqueeze(-1),
                                                                combined_scores.unsqueeze(-1))
            copy_log_probs_cpu = copy_log_probs.cpu()
            if i < (trimmed_source_length - 1):
                source_future_occurences = (source_token_ids[:, (i+1):] == source_token_ids[:, i].unsqueeze(-1)).float()  # pylint: disable=line-too-long
                future_copy_log_probs = copy_log_probs[:, (i+1):] + (source_future_occurences + 1e-45).log()
                # shape: (group_size, 1 + trimmed_source_length - i)
                combined = torch.cat((copy_log_probs_slice.unsqueeze(-1), future_copy_log_probs), dim=-1)
                # shape: (group_size,)
                copy_log_probs_slice = util.logsumexp(combined)
            if i > 0:
                source_previous_occurences = source_token_ids[:, 0:i] == source_token_ids[:, i].unsqueeze(-1)
                duplicate_mask = (source_previous_occurences.sum(dim=-1) == 0).float()
                copy_log_probs_slice = copy_log_probs_slice + (duplicate_mask + 1e-45).log()

            left_over_copy_log_probs = copy_log_probs_slice + (1.0 - copy_log_probs_to_add_mask + 1e-45).log()
            modified_log_probs_list.append(left_over_copy_log_probs.unsqueeze(-1))
        modified_log_probs_list.insert(0, generation_log_probs)
        modified_log_probs = torch.cat(modified_log_probs_list, dim=-1)

        return modified_log_probs

    
    def _decoder_step(self, last_predictions, selective_weights, state):
        bz, msl, hs = state["encoder_hs"].size()
        ngt = self.ngt
        nt = self._decoder_layers

        embedded_input = self.dropout(self.LayerNorm(self._target_embedder(last_predictions)))
        #embedded_input = self._target_embedder(last_predictions)

        encoder_extended_attention_mask = (1.0 - state["source_mask"][:, None, None, :]) * -10000.0
        cross_attention_outputs = self._attention(hidden_states=state["decoder_hidden"].view(bz, 1, hs), 
                encoder_hidden_states=state["encoder_hs"], 
                encoder_attention_mask=encoder_extended_attention_mask)
        attentive_read = cross_attention_outputs[0].view(bz, -1)  # [bz * ngt * cur_ngt, hs]
        # attentive_weights = self._attention(state["decoder_hidden"], state["encoder_hs"], state["source_mask"].float())
        # attentive_read = util.weighted_sum(state["encoder_hs"], attentive_weights)  # [bz * ngt, hs]

        selective_read = util.weighted_sum(state["encoder_hs"][:, 1:-1], selective_weights)
        
        decoder_input = torch.cat((embedded_input, attentive_read, selective_read, state["set_hs"]), -1)
        #decoder_input = torch.cat((embedded_input, attentive_read, selective_read), -1)
        projected_decoder_input = self._input_projection_layer(decoder_input) # [bz * ngt, hs]
        
        state["decoder_hidden_all"] = state["decoder_hidden_all"].view(-1, nt, hs).transpose(0,1).contiguous()
        state["decoder_context_all"] = state["decoder_context_all"].view(-1, nt, hs).transpose(0,1).contiguous()
        _, (state["decoder_hidden_all"], state["decoder_context_all"]) = self._decoder_cell(projected_decoder_input.unsqueeze(1), (state["decoder_hidden_all"], state["decoder_context_all"]))

        state["decoder_hidden"] = self._decoder_dropout(state["decoder_hidden_all"][-1]) # [bz * ngt, hs]
        state["decoder_context"] = state["decoder_context_all"][-1]
        state["decoder_hidden_all"] = state["decoder_hidden_all"].transpose(0,1).contiguous().view(-1, nt * hs)
        state["decoder_context_all"] = state["decoder_context_all"].transpose(0,1).contiguous().view(-1, nt * hs)
        return state


    def _get_predicted_tokens(self, predicted_indices, batch_metadata, n_best):
        """
        Convert predicted indices into tokens.
        If `n_best = 1`, the result type will be `List[List[str]]`. Otherwise the result
        type will be `List[List[List[str]]]`.
        """
        if not isinstance(predicted_indices, np.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        predicted_tokens = []
        # predicted_indices.shape  [bz, 1, ngt * max_decoding_steps]
        for top_k_predictions, metadata in zip(predicted_indices, batch_metadata):
            batch_predicted_tokens = []
            for indices in top_k_predictions[:n_best]:
                tokens: List[str] = []
                indices = list(indices)
                # if self._eoe_index in indices:  # will not be true if max number of extractions already reached
                #     indices = indices[:indices.index(self._eoe_index)]
                indices = self._leave_one(indices, self._end_index)
                for index in indices:
                    if index >= self._target_vocab_size:
                        adjusted_index = index - self._target_vocab_size
                        token = metadata["source_tokens"][adjusted_index]
                    else:
                        token = self.vocab.get_token_from_index(index, self._target_namespace)
                    tokens.append(token)
                batch_predicted_tokens.append(tokens)
            if n_best == 1:
                predicted_tokens.append(batch_predicted_tokens[0])
            else:
                predicted_tokens.append(batch_predicted_tokens)
        return predicted_tokens

    def calculate_loss_weight(self, vector):
        vector = vector.float()
        num0 = (1 - vector).sum()
        num1 = vector.sum()
        weight0 = num1 / (num0 + num1)
        weight1 = num0 / (num0 + num1)
        return torch.Tensor([weight0, weight1]).to(vector.device)


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Finalize predictions.
        After a beam search, the predicted indices correspond to tokens in the target vocabulary
        OR tokens in source sentence. Here we gather the actual tokens corresponding to
        the indices.
        """
        predicted_tokens = self._get_predicted_tokens(output_dict["predictions"],
                                                      output_dict["metadata"])
        output_dict["predicted_tokens"] = predicted_tokens
        return output_dict


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(self._tensor_based_metric.get_metric(reset=reset))  # type: ignore
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

class SetDecoder(nn.Module):
    def __init__(self, config, ngt, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(ngt, config.hidden_size)
        self.query_projection_layer = nn.Linear(config.hidden_size, config.hidden_size)

        self.query_output_layer = nn.Linear(config.hidden_size, 2)
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, state):
        bz = state["encoder_hs"].size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bz, 1, 1)  # [bz, ngt, hs]
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(
                hidden_states, state["encoder_hs"], state["source_mask"]
            )
            hidden_states = layer_outputs[0]
        
        set_hidden_state = self.query_projection_layer(hidden_states)
        set_pred_mask = self.query_output_layer(set_hidden_state)
        state["set_hs"] =  set_hidden_state  #  [bz, ngt, hs]
        return state, set_pred_mask


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs