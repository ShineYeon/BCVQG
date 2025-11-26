from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
from torch.nn.functional import softmax


class BLIP_VQA(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=480,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer,
                                                       drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

    def forward(self, image, question, answer=None, n=None, weights=None, train=True, inference='rank', k_test=128):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35,
                                  return_tensors="pt").to(image.device)
        question.input_ids[:, 0] = self.tokenizer.enc_token_id

        if train:
            '''
            n: number of answers for each question
            weights: weight for each answer
            '''
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device)
            answer.input_ids[:, 0] = self.tokenizer.bos_token_id
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)

            question_states = []
            question_atts = []
            for b, n in enumerate(n):
                question_states += [question_output.last_hidden_state[b]] * n
                question_atts += [question.attention_mask[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=question_states,
                                              encoder_attention_mask=question_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              )

            loss = weights * answer_output.loss
            loss = loss.sum() / image.size(0)

            return loss


        else:
            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)

            if inference == 'generate':
                num_beams = 1
                question_states = question_output.last_hidden_state.repeat_interleave(num_beams, dim=0)
                question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long).to(question_states.device)
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask": question_atts}

                # 정답 및 오답 생성
                correct_answer, wrong_answers = self.generate_answer_and_wrong_answers(
                    model_kwargs, image.device, max_length=20, top_n=10
                )

                return {"correct_answer": correct_answer, "wrong_answers": wrong_answers}



            elif inference == 'rank':
                max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask,
                                           answer.input_ids, answer.attention_mask, k_test)
                return max_ids

    def generate_answer_and_wrong_answers(self, model_kwargs, device, max_length=20, top_n=3):
        """
        Generate the correct answer and wrong answers based on token probabilities.

        Args:
            model_kwargs (dict): Model kwargs for generation.
            device (torch.device): Device for computation.
            max_length (int): Maximum length of generated sequences.
            top_n (int): Number of wrong answers to generate (e.g., 2 for top 4, 5).

        Returns:
            tuple: Correct answer string and list of wrong answer strings.
        """
        bos_ids = torch.full((1, 1), fill_value=self.tokenizer.bos_token_id, device=device)
        input_ids = bos_ids
        correct_tokens = []  # 정답 토큰 ID를 저장할 리스트
        wrong_answers = []

        for step in range(max_length):
            outputs = self.text_decoder(
                input_ids=input_ids,
                **model_kwargs,
                return_dict=True,
                output_attentions=False,
            )
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            probs = softmax(logits, dim=-1)

            # 정답 토큰 선택
            top_token_id = torch.argmax(probs, dim=-1).item()  # .item()으로 단일 정수 ID 추출
            if top_token_id == self.tokenizer.eos_token_id or top_token_id == self.tokenizer.sep_token_id:
                break  # 종료 조건: EOS 또는 SEP 토큰
            correct_tokens.append(top_token_id)  # 정수 ID를 리스트에 추가

            # 정답 제외 후 top 4, 5 토큰 선택
            probs[0, top_token_id] = -1  # 정답 제외
            sorted_indices = torch.argsort(probs, descending=True).flatten()
            top_wrong_ids = sorted_indices[2:12].tolist()  # 4번째, 5번째 선택
            wrong_tokens = [self.tokenizer.decode([token_id]).strip() for token_id in top_wrong_ids]

            # 정답 토큰을 입력에 추가
            input_ids = torch.cat([input_ids, torch.tensor([[top_token_id]], device=device)], dim=-1)

            # 오답 리스트 업데이트
            wrong_answers.extend(wrong_tokens)

        # 정답 디코딩
        correct_answer = self.tokenizer.decode(correct_tokens, skip_special_tokens=True).strip()

        # 오답 정리
        wrong_answers = list(set(wrong_answers))[:top_n]  # 중복 제거 및 개수 제한

        return correct_answer, wrong_answers

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques, k)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]

        return max_ids


def blip_vqa(pretrained='', **kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    #         assert(len(msg.missing_keys)==0)
    return model


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))

