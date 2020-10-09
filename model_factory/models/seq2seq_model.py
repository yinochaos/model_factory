#!/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2020 yinochaos <pspcxl@163.com>. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tqdm
from typing import Any, Tuple, List, Dict
import tensorflow as tf
import numpy as np
from model_factory.utils import BeamHypotheses
from model_factory.models.model import Model
from model_factory.losses.loss import seq2seq_cross_entropy_loss
from datasets.utils.common_struct import data_schemas2types, data_schemas2shapes

""" model interface
"""


class Seq2seqModel(Model):
    def __init__(self, optimizer, loss, encoder, decoder, max_decoder_len=16, vocab_size=None, feature_fields=None, label_fields=None, batch_size=16):
        self.optimizer = optimizer
        self.loss = loss
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_size = encoder.hidden_size
        self.max_decoder_len = max_decoder_len
        self.vocab_size = vocab_size or decoder.vocab_size()
        self._build_model()
        # self.model.summary()

    def _build_model(self):
        inputs = self.encoder.get_inputs()
        _, enc_hidden = self.encoder(inputs)
        tokens_inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='tokens')
        logits, _ = self.decoder([tokens_inputs, enc_hidden])
        self.model = tf.keras.Model([inputs, tokens_inputs], logits)
        #loss1=KL.Lambda(lambda x:custom_loss1(*x),name='loss1')([x,x_in])
        loss = tf.keras.layers.Lambda(lambda x: seq2seq_cross_entropy_loss(x[0], x[1]))([tokens_inputs[1:], logits[:-1]])
        self.model.add_loss(loss)
        self.model.compile(optimizer=self.optimizer)

    def predict_beam(self, input_ids, max_length=32, early_stopping=False, num_beams=10, temperature=1.0,
                     top_k=50, repetition_penalty=1.0, pad_token_id=0, bos_token_id=1, eos_token_id=2, length_penalty=1.0,
                     no_repeat_ngram_size=0, bad_words_ids=None, num_return_sequences=1, use_attention_mask=False, attention_mask=None, use_cache=True):
        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
        assert temperature > 0, "`temperature` should be strictely positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictely positive."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictely positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"
        batch_size = input_ids.shape[0]
        effective_batch_size = batch_size
        effective_batch_mult = 1

        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask) if use_attention_mask else self.encoder(input_ids)
        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = tf.broadcast_to(tf.expand_dims(input_ids, 1), (batch_size, effective_batch_mult * num_beams, input_ids_len))
            if use_attention_mask:
                attention_mask = tf.broadcast_to(tf.expand_dims(attention_mask, 1), (batch_size, effective_batch_mult * num_beams, input_ids_len))
                # shape: (batch_size * num_return_sequences * num_beams, cur_len)
                attention_mask = tf.reshape(attention_mask, (effective_batch_size * num_beams, input_ids_len))
            # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            input_ids = tf.reshape(input_ids, (effective_batch_size * num_beams, input_ids_len))
        # create empty decoder_input_ids
        input_ids = (tf.ones((effective_batch_size * num_beams, 1), dtype=tf.int32,) * bos_token_id)
        assert (
            batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "
        expanded_batch_idxs = tf.reshape(
            tf.repeat(tf.expand_dims(tf.range(batch_size), -1), repeats=num_beams * effective_batch_mult, axis=1),
            shape=(-1,),
        )
        # expand encoder_outputs
        #encoder_outputs = (tf.gather(encoder_outputs[0], expanded_batch_idxs, axis=0), *encoder_outputs[1:])
        encoder_outputs = tf.gather(encoder_outputs[1], expanded_batch_idxs, axis=0)
        return self._generate_beam_search(input_ids, 1, max_length, early_stopping, temperature, top_k, repetition_penalty,
                                          no_repeat_ngram_size, bad_words_ids, pad_token_id, eos_token_id, batch_size, num_return_sequences, length_penalty, num_beams,
                                          self.vocab_size, encoder_outputs, use_attention_mask, attention_mask, use_cache)

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        early_stopping,
        temperature,
        top_k,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        use_attention_mask,
        attention_mask,
        use_cache,
    ):
        """Generate sequences for each example with beam search."""

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        beam_scores_begin = tf.zeros((batch_size, 1), dtype=tf.float32)
        beam_scores_end = tf.ones((batch_size, num_beams - 1), dtype=tf.float32) * (-1e9)
        beam_scores = tf.concat([beam_scores_begin, beam_scores_end], -1)

        beam_scores = tf.reshape(beam_scores, (batch_size * num_beams,))

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            outputs = self.decoder([input_ids, encoder_outputs])  # (B * num_beams, cur_len, V)
            next_token_logits = outputs[0][:, -1, :]  # (B * num_beams, V)
            # calculate log softmax score
            scores = tf.nn.log_softmax(next_token_logits, axis=-1)  # (B * num_beams, V)

            assert scores.shape == [batch_size * num_beams, vocab_size]

            # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
            next_scores = scores + tf.broadcast_to(beam_scores[:, None], (batch_size * num_beams, vocab_size))  # (B * num_beams, V)
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = tf.reshape(next_scores, (batch_size, num_beams * vocab_size))  # (batch_size, num_beams * vocab_size)
            next_scores, next_tokens = tf.math.top_k(next_scores, k=2 * num_beams, sorted=True)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence or last iteration
                    if (eos_token_id is not None) and (token_id.numpy() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            tf.identity(input_ids[effective_beam_id]), beam_token_score.numpy()
                        )
                    else:
                        # add next predicted token if it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if we are done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    tf.reduce_max(next_scores[batch_idx]).numpy(), cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1)

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = tf.convert_to_tensor([x[0] for x in next_batch_beam], dtype=tf.float32)
            beam_tokens = tf.convert_to_tensor([x[1] for x in next_batch_beam], dtype=tf.int32)
            beam_idx = tf.convert_to_tensor([x[2] for x in next_batch_beam], dtype=tf.int32)

            # re-order batch and update current length
            input_ids = tf.stack([tf.identity(input_ids[x, :]) for x in beam_idx])
            input_ids = tf.concat([input_ids, tf.expand_dims(beam_tokens, 1)], axis=-1)
            cur_len = cur_len + 1

        # finalize all open beam hypotheses and end to generated hypotheses
        for batch_idx in range(batch_size):
            # Add all open beam hypothesis to generated_hyps
            if done[batch_idx]:
                continue
            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).numpy().item() != eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert tf.reduce_all(
                    next_scores[batch_idx, :num_beams] == tf.reshape(beam_scores, (batch_size, num_beams))[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx], tf.reshape(beam_scores, (batch_size, num_beams))[batch_idx]
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].numpy().item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size * num_return_sequences
        output_num_return_sequences_per_batch = num_return_sequences

        # select the best hypotheses
        sent_lengths_list = []
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths_list.append(len(best_hyp))
                best.append(best_hyp)
        assert output_batch_size == len(best), "Output batch size {} must match output beam hypotheses {}".format(output_batch_size, len(best))

        sent_lengths = tf.convert_to_tensor(sent_lengths_list, dtype=tf.int32)

        # shorter batches are filled with pad_token
        if tf.reduce_min(sent_lengths).numpy() != tf.reduce_max(sent_lengths).numpy():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(tf.reduce_max(sent_lengths).numpy() + 1, max_length)
            decoded_list = []

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                assert sent_lengths[i] == tf.shape(hypo)[0]
                # if sent_length is max_len do not pad
                if sent_lengths[i] == sent_max_len:
                    decoded_slice = hypo
                else:
                    # else pad to sent_max_len
                    num_pad_tokens = sent_max_len - sent_lengths[i]
                    padding = pad_token_id * tf.ones((num_pad_tokens,), dtype=tf.int32)
                    decoded_slice = tf.concat([hypo, padding], axis=-1)

                    # finish sentence with EOS token
                    if sent_lengths[i] < max_length:
                        decoded_slice = tf.where(
                            tf.range(sent_max_len, dtype=tf.int32) == sent_lengths[i],
                            eos_token_id * tf.ones((sent_max_len,), dtype=tf.int32),
                            decoded_slice,
                        )
                # add to list
                decoded_list.append(decoded_slice)

            decoded = tf.stack(decoded_list)
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = tf.stack(best)

        return decoded

    def predict(self, inputs, bos_token_id=None, eos_token_id=None, token_dict=None):
        def id_to_tokens(ids, token_dict):
            return [token_dict[id] if id in token_dict else '<UNKOWN>' for id in ids]
        results = []
        #attentions = []

        bos_token_id = 1 if bos_token_id is None else bos_token_id
        eos_token_id = 2 if eos_token_id is None else eos_token_id

        _, enc_hidden = self.encoder(inputs)
        dec_input = tf.expand_dims([bos_token_id] * inputs.shape[0], 1)
        dec_input = tf.cast(dec_input, dtype=tf.int64)
        print('dec    inputs', dec_input)
        for _ in range(self.max_decoder_len):
            predictions, _ = self.decoder([dec_input, enc_hidden])  # (batch_size, cur_len, vocab_size)
            print('predictions', predictions.shape)
            next_token_logits = predictions[:, -1, :]  # (batch_size, vocab_size)
            next_token_logits = tf.reshape(next_token_logits, [inputs.shape[0], -1])
            print('predictions', next_token_logits.shape)
            next_tokens = tf.argmax(next_token_logits, axis=-1)
            results.append(next_tokens.numpy())
            #print('dec_inputs', dec_input.shape)
            dec_input = tf.concat([dec_input, tf.expand_dims(next_tokens, 1)], axis=-1)
            #input_ids = tf.concat([input_ids, tf.expand_dims(beam_tokens, 1)], axis=-1)
            print('next_token', next_tokens.numpy(), 'dec_input', dec_input.numpy())
        return results

    # here tf.function can fix error: tensorflow.python.framework.errors_impl.UnknownError: CUDNN_STATUS_BAD_PARAM
    # in tensorflow/stream_executor/cuda/cuda_dnn.cc(1521):
    # 'cudnnSetRNNDataDescriptor( data_desc.get(), data_type, layout,
    # max_seq_length, batch_size, data_size, seq_lengths_array,
    # (void*)&padding_fill)' [Op:CudnnRNNV3]

    def train_step(self, inputs, targets):
        return self.model.train_on_batch([inputs, targets], y=None)
