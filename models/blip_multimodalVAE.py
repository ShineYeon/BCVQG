from models.med import BertConfig, BertModel, BertLMHeadModelForLatentInjection
from models.blip import create_vit, load_checkpoint
from models.CVAE import AverageSelfAttention

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary


class BLIP_VQG(nn.Module):
    def __init__(self,
                 tokenizer,
                 med_config='configs/med_config.json',
                 image_size=480,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 add_type_to_posterior=False,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer,
                                                       drop_path_rate=0.1) #vision width=768
        self.tokenizer = tokenizer
        text_encoder_config = BertConfig.from_json_file(med_config)
        self.text_encoder_for_cross = BertModel(config=text_encoder_config, add_pooling_layer=False)

        self.averageSelfAttention = AverageSelfAttention(text_encoder_config.encoder_width)
        nx = text_encoder_config.encoder_width
        nz = text_encoder_config.encoder_width

        self.attn_proj = nn.Linear(nz, nx, bias=False)

        self.mean = Conv1D(nz, nx)
        self.logvar = Conv1D(nz, nx)

        n = text_encoder_config.num_hidden_layers

        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModelForLatentInjection(config=decoder_config)

        self.add_type_to_posterior = add_type_to_posterior


        # For Phase 1: Train Encoder Only
    def freeze_decoder(self):
        for param in self.averageSelfAttention.parameters():
            param.requires_grad = False
        for param in self.attn_proj.parameters():
            param.requires_grad = False
        for param in self.mean.parameters():
            param.requires_grad = False
        for param in self.logvar.parameters():
            param.requires_grad = False
        for param in self.text_decoder.parameters():
            param.requires_grad = False

    # Latent variable 'z' 샘플링하는 데 사용됨.
    # VAE에서는 latent variable 'z'를 mean과 log variance를 사용해 정규분포로부터 샘플링함.
    def reparameterize(self, mean, logvar, z=None):
        # std = exp(logvar/2), 즉, logvar을 사용해 표준편차 std 계산
        std = logvar.mul(0.5).exp()  # log var의 값을 0.5배(1/2) 후 exp() 취하면 표준 편차 std 얻음.
        if z is None:  # z가 주어지지 않은 경우 표준 정규 분포에서 샘플링
            z = torch.randn(std.size(), device=mean.device, dtype=mean.dtype)
            # 평균이 0이고 표준편차가 1인 정규분포에서 샘플링(torch.randn이 평균 0, variance 1인 normal distribution으로부터 샘플링하여 랜덤 숫자로 채워진 텐서 반환)
        return z.mul(std) + mean
        # z = \mu + \sigma * \epsilon, 즉 표준 정규 분포에서 샘플링된 값을 std와 곱하고 mean을 더해 최종 잠재변수 z 생성
        # \epsilon은 샘플링 과정에서 무작위성을 도입하는 것 => 모델이 다양한 Latent space에서 샘플을 생성할 수 있게 하는 것.

    # 두 정규분포 간 KL-divergence 계산
    # VAE에서 인코더가 생성한 latent variable의 distribution이 prior distribution과 얼마나 다른지 측정
    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()  # 배치차원에 대해 평균 계산 => 최종 KL-divergence 값 반환

    def forward(self, beta, image, question=None, answer=None, category=None, qlength=None, alength=None, train=True,
                inference='rank', k_test=128, top_k=50, top_p=0.95, temperature=1.0, repetition_penalty=1.2, num_beams=3):

        #print(f"Type of image in forward before visual_encoder: {type(image)}")

        # 이미지 -> 임베딩
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # C to VAE
        answer_type = self.tokenizer(category, padding='longest', truncation=True, max_length=4, return_tensors='pt').to(image.device)
        answer_type.input_ids[:, 0] = self.tokenizer.enc_token_id # [ENC] token

        if train:
            # TODO: 이미지+답변유형 => Prior, 이미지+질문 => Posterior 인 VAE 구현
            type_with_image = self.text_encoder_for_cross(answer_type.input_ids,
                                                          attention_mask=answer_type.attention_mask,
                                                          encoder_hidden_states=image_embeds,
                                                          encoder_attention_mask=image_atts,
                                                          return_dict=True)
            type_with_image_states = type_with_image.last_hidden_state
            type_with_image_atts = torch.ones(type_with_image_states.size()[:-1],dtype=torch.long).to(type_with_image_states.device)

            ques_tokens = self.tokenizer(question, padding='longest', truncation=True, max_length=35, return_tensors='pt').to(image.device)
            ques_tokens.input_ids[:, 0] = self.tokenizer.bos_token_id # 맨 앞 [CLS] 토큰 -> [DEC] 토큰
            ques_with_image = self.text_encoder_for_cross(ques_tokens.input_ids,
                                                          attention_mask=ques_tokens.attention_mask,
                                                          encoder_hidden_states=image_embeds,
                                                          encoder_attention_mask=image_atts,
                                                          return_dict=True)
            ques_with_image_states = ques_with_image.last_hidden_state
            ques_with_image_atts = torch.ones(ques_with_image_states.size()[:-1], dtype=torch.long).to(ques_with_image_states.device)


            prior_representations, _ = self.averageSelfAttention(type_with_image_states, type_with_image_atts)
            prior_mean = self.mean(prior_representations)
            prior_logvar = self.logvar(prior_representations)


            # Posterior: x to VAE
            # print(category)

            ##### ==================== [ENC]type[DEC]question ==================== #####
            category_tokens = self.tokenizer(category, truncation=True, padding='longest', max_length=4,
                                             return_tensors='pt').to(image.device)  # 'predicate'이 2개로 쪼개짐.
            # print(category_tokens)
            category_tokens.input_ids = category_tokens.input_ids[:, :-1]  # 마지막 [SEP] 토큰 삭제
            category_tokens.token_type_ids = category_tokens.token_type_ids[:, :-1]
            category_tokens.attention_mask = category_tokens.attention_mask[:, :-1]
            category_tokens.input_ids[:, 0] = self.tokenizer.enc_token_id  # 맨앞 [CLS] 토큰 -> [ENC] 토큰 대체
            question_tokens = self.tokenizer(question, padding='longest', truncation=True, max_length=35,
                                             return_tensors='pt').to(image.device)
            question_tokens.input_ids[:, 0] = self.tokenizer.bos_token_id  # Question 맨 앞 [CLS] -> [DEC] 토큰

            # type + question 결합: combined_input_ids와 combined_attention_mask 리스트 초기화
            combined_input_ids = []
            combined_token_type_ids = []
            combined_attention_mask = []

            # type + question 결합: 배치 내 각 샘플에 대해 처리
            for i in range(category_tokens.input_ids.size(0)):
                # question 토크나이즈 결과 결합
                combined_ids = torch.cat([category_tokens.input_ids[i], question_tokens.input_ids[i]])
                combined_type_ids = torch.cat(
                    [category_tokens.token_type_ids[i], question_tokens.token_type_ids[i]])
                combined_mask = torch.cat([category_tokens.attention_mask[i], question_tokens.attention_mask[i]])

                combined_input_ids.append(combined_ids)
                combined_token_type_ids.append(combined_type_ids)
                combined_attention_mask.append(combined_mask)

            question_with_type = {
                'input_ids': torch.stack(combined_input_ids),
                'token_type_ids': torch.stack(combined_token_type_ids),
                'attention_mask': torch.stack(combined_attention_mask)
            }
            question_with_type = {k: v.to(image.device) for k, v in question_with_type.items()}
            ##### ==================== [ENC]type[DEC]question ==================== #####


            # TODO: 뭔가 꼬여있음.. posterior를 이미지+(답변+질문)으로 할지, 이미지+(질문)으로 할지 결정필요. => 일단 이미지+(질문)으로 실험
            if self.add_type_to_posterior: # Posterior가 이미지+(답변+질문)
                question_output = self.text_encoder_for_cross(question_with_type['input_ids'],
                                                              attention_mask=question_with_type['attention_mask'],
                                                              encoder_hidden_states=image_embeds,
                                                              encoder_attention_mask=image_atts,
                                                              return_dict=True)
                question_states = question_output.last_hidden_state
                question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long).to(question_states.device)
                posterior_representations, _ = self.averageSelfAttention(question_states, question_atts)
                posterior_mean = self.mean(posterior_representations)
                posterior_logvar = self.logvar(posterior_representations)

            else: # Posterior가 이미지+(질문)
                ques_tokens = self.tokenizer(question, padding='longest', truncation=True, max_length=35,
                                             return_tensors='pt').to(image.device)
                ques_tokens.input_ids[:, 0] = self.tokenizer.bos_token_id  # 맨 앞 [CLS] 토큰 -> [DEC] 토큰
                ques_with_image = self.text_encoder_for_cross(ques_tokens.input_ids,
                                                              attention_mask=ques_tokens.attention_mask,
                                                              encoder_hidden_states=image_embeds,
                                                              encoder_attention_mask=image_atts,
                                                              return_dict=True)
                ques_with_image_states = ques_with_image.last_hidden_state
                ques_with_image_atts = torch.ones(ques_with_image_states.size()[:-1], dtype=torch.long).to(
                    ques_with_image_states.device)
                posterior_representations, _ = self.averageSelfAttention(ques_with_image_states, ques_with_image_atts)
                posterior_mean = self.mean(posterior_representations)
                posterior_logvar = self.logvar(posterior_representations)

            ##### VAE
            latent_mean, latent_logvar = posterior_mean, posterior_logvar
            z = self.reparameterize(latent_mean, latent_logvar)
            assert not torch.isnan(z).any(), 'training get nan z'

            attn_proj = self.attn_proj(z).unsqueeze(1)

            question_targets = question_with_type['input_ids'].masked_fill(question_with_type['input_ids'] == self.tokenizer.pad_token_id, -100)

            decoder_output = self.text_decoder(question_with_type['input_ids'],
                                               attention_mask=question_with_type['attention_mask'],
                                               encoder_hidden_states=image_embeds,
                                               encoder_attention_mask=image_atts,
                                               labels=question_targets,
                                               return_dict=True,
                                               reduction='none',
                                               z=attn_proj)

            lm_loss = decoder_output.loss.mean()

            kl_loss = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar)

            loss = lm_loss + beta * kl_loss

            return loss, lm_loss, kl_loss


        # TODO: [ENC] + type -> [ENC] + type + [BOS] + question
        else:
            type_with_image = self.text_encoder_for_cross(answer_type.input_ids,
                                                          attention_mask=answer_type.attention_mask,
                                                          encoder_hidden_states=image_embeds,
                                                          encoder_attention_mask=image_atts,
                                                          return_dict=True)
            type_with_image_states = type_with_image.last_hidden_state
            type_with_image_atts = torch.ones(type_with_image_states.size()[:-1], dtype=torch.long).to(
                type_with_image_states.device)

            prior_representations, _ = self.averageSelfAttention(type_with_image_states, type_with_image_atts)

            try:
                prior_mean = self.mean(prior_representations)
                prior_logvar = self.logvar(prior_representations)
            except:
                prior_mean, prior_logvar = torch.zeros([image.size(0), self.n_embed], device=image.device)

            latent_mean, latent_logvar = prior_mean, prior_logvar
            z = self.reparameterize(latent_mean, latent_logvar)
            assert not torch.isnan(z).any(), 'training get nan z'
            attn_proj = self.attn_proj(z).unsqueeze(1)

            model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts, "z": attn_proj}

            category_tokens = self.tokenizer(category, truncation=True, padding='longest', max_length=4,
                                             return_tensors='pt').to(image.device)
            category_tokens.input_ids[:, 0] = self.tokenizer.enc_token_id
            category_tokens.input_ids[:, -1] = self.tokenizer.bos_token_id

            outputs = self.text_decoder.generate(input_ids=category_tokens.input_ids,
                                                     max_length=35,
                                                     min_length=5,
                                                     num_beams=num_beams,
                                                     do_sample=True,
                                                     top_p=top_p,
                                                     temperature=temperature,
                                                     repetition_penalty=repetition_penalty,
                                                     eos_token_id=self.tokenizer.sep_token_id,
                                                     pad_token_id=self.tokenizer.pad_token_id,
                                                     **model_kwargs)

            res_questions = []
            for output in outputs:
                # [DEC] 토큰 이후의 토큰만 추출
                bos_token_indices = (output == self.tokenizer.bos_token_id).nonzero(as_tuple=True)
                if len(bos_token_indices[0]) > 0:
                    dec_token_index = bos_token_indices[0][0].item()
                    question_tokens = output[dec_token_index + 1:]
                else:
                    # [DEC] 토큰이 없는 경우 전체 시퀀스를 사용
                    question_tokens = output

                # 질문 토큰을 문자열로 디코딩
                res_question = self.tokenizer.decode(question_tokens, skip_special_tokens=True)
                res_questions.append(res_question)

            return res_questions


def blip_vqg2(tokenizer, pretrained='', **kwargs):
    model = BLIP_VQG(tokenizer, **kwargs)
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