from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class BLIP_VQG(nn.Module):
    def __init__(self,
                 tokenizer,
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
        self.tokenizer = tokenizer

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

    def forward(self, image, question=None, answer=None, category=None, qlength=None, alength=None, train=True,
                inference='rank', k_test=128, top_p=0.95, temperature=1.0, repetition_penalty=1.2, num_beams=3):

        # 이미지 -> 임베딩
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        answer_category = self.tokenizer(category, padding='longest', truncation=True, max_length=35,
                                return_tensors='pt').to(image.device)
        answer_category.input_ids[:, 0] = self.tokenizer.enc_token_id

        if train:
            # 이미지를 encoder hidden state로 넣어서 질문 생성 모델링, loss는 language modeling loss
            # 생성해야될 ground truth 질문 인코딩
            # h5py 파일에 있는 question은 인코딩된 상태였지만 data_loader.py에서 다시 디코딩해 문자열로 만들어줬음.

            # print(question)
            question = self.tokenizer(question, padding='longest', return_tensors="pt").to(image.device)

            question.input_ids[:, 0] = self.tokenizer.bos_token_id

            question_targets = question.input_ids.masked_fill(question.input_ids == self.tokenizer.pad_token_id, -100)

            answer_category_output = self.text_encoder(answer_category.input_ids,
                                              attention_mask=answer_category.attention_mask,
                                              encoder_hidden_states=image_embeds,
                                              encoder_attention_mask=image_atts,
                                              return_dict=True)

            answer_category_states = answer_category_output.last_hidden_state
            answer_category_atts = answer_category.attention_mask

            question_output = self.text_decoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=answer_category_states,
                                                encoder_attention_mask=answer_category_atts,
                                                labels=question_targets,
                                                return_dict=True,
                                                reduction='none')
            loss = question_output.loss

            return loss
        else:
            answer_category_output = self.text_encoder(answer_category.input_ids,
                                              attention_mask=answer_category.attention_mask,
                                              encoder_hidden_states=image_embeds,
                                              encoder_attention_mask=image_atts,
                                              return_dict=True)
            # 이미지를 encoder hidden state로 넣어서 질문 생성 후 생성된 질문 반환
            num_beams = 3
            answer_category_states = answer_category_output.last_hidden_state
            answer_category_atts = torch.ones(answer_category_states.size()[:-1], dtype=torch.long).to(answer_category_states.device)
            model_kwargs = {"encoder_hidden_states": answer_category_states, "encoder_attention_mask": answer_category_atts}

            bos_ids = torch.full((image.size(0), 1), fill_value=self.tokenizer.bos_token_id, device=image.device)

            outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                 max_length=30,
                                                 min_length=5,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 temperature=temperature,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 **model_kwargs)
            res_questions = []
            for output in outputs:
                res_question = self.tokenizer.decode(output, skip_special_tokens=True)
                res_questions.append(res_question)
            return res_questions




def blip_vqg_base(tokenizer, pretrained='', **kwargs):
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

