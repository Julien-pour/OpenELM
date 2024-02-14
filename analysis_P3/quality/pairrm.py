import torch
import torch.nn as nn

from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    SequenceClassifierOutput
)
from typing import Optional, Tuple, Union
    
class DebertaV2PairRM(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.n_tasks = config.n_tasks
        self.drop_out = config.drop_out
        
        # LM
        self.pretrained_model = DebertaV2Model(config)
        self.hidden_size = config.hidden_size
        
        self.sep_token_id = config.sep_token_id # to add
        self.source_prefix_id = config.source_prefix_id # to add
        self.cand_prefix_id = config.cand_prefix_id
        self.cand1_prefix_id = config.cand1_prefix_id
        self.cand2_prefix_id = config.cand2_prefix_id
        
        self.head_layer = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(2*self.hidden_size, 1*self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.drop_out),
            nn.Linear(1 * self.hidden_size, self.n_tasks),
        )
        self.sigmoid = nn.Sigmoid()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        #  <source_prefix_id>...<sep><cand1_prefix_id>...<sep><cand2_prefix_id> ... <sep>
        assert all([self.source_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]), "<source> id not in input_ids"
        assert all([self.cand1_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]), "<candidate1> id not in input_ids"
        assert all([self.cand2_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]), "<candidate2> id not in input_ids"
        
        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        encs = outputs.hidden_states[-1]
        source_idxs = torch.where(input_ids == self.source_prefix_id)
        source_encs = encs[source_idxs[0], source_idxs[1], :]
        cand1_idxs = torch.where(input_ids == self.cand1_prefix_id)
        cand1_encs = encs[cand1_idxs[0], cand1_idxs[1], :]
        cand2_idxs = torch.where(input_ids == self.cand2_prefix_id)
        cand2_encs = encs[cand2_idxs[0], cand2_idxs[1], :]
        
        # reduce
        source_cand1_encs = torch.cat([source_encs, cand1_encs], dim=-1)
        source_cand2_encs = torch.cat([source_encs, cand2_encs], dim=-1)
        left_pred_scores = self.head_layer(source_cand1_encs)
        right_pred_scores = self.head_layer(source_cand2_encs)

        loss = None
        if labels is not None:
            loss = self.compute_loss(left_pred_scores, right_pred_scores, labels)
        

        preds = (left_pred_scores - right_pred_scores).mean(dim=-1)
        return SequenceClassifierOutput(
            loss=loss, logits=preds, 
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions
        )
    
    def compute_loss(self, left_pred_scores, right_pred_scores, labels):
        """
        Args:
            left_pred_scores: [n_candidates, n_task]
            right_pred_scores: [n_candidates, n_task]
            labels: [n_candidates, n_task], 1/0/-1 for left/right/both is better
        """

        device = left_pred_scores.device
        loss = torch.tensor(0.0).to(left_pred_scores.device)
        
        dif_scores = labels
        left_pred_scores = left_pred_scores * dif_scores.sign()
        right_pred_scores = - right_pred_scores * dif_scores.sign()
        cls_loss = torch.tensor(0.0, device=device)
        cls_loss += - torch.log(torch.sigmoid(left_pred_scores+right_pred_scores)).mean()
        loss += cls_loss
        return loss
    


import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

## Define the reward model function class

class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return scores

# utils for GAIR/autoj-13b-GPTQ-4bits
    
PROMPT_INPUT_SYSTEM: str = '[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{input} [/INST]'

PROMPT_INPUT_WO_SYSTEM: str = "[INST] {input} [/INST]"

PROMPT_INPUT_FOR_SCENARIO_CLS: str = "Identify the scenario for the user's query, output 'default' if you are uncertain.\nQuery:\n{input}\nScenario:\n"

single = """Write critiques for a submitted response on a given user's query, and grade the response:
  
[BEGIN DATA]
***
[Query]: {prompt}
***
[Response]: {response}
***
[END DATA]

Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"."""

pairwise_tie = """You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {prompt}
***
[Response 1]: {response}
***
[Response 2]: {response_another}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided."""

protocol_mapping = {
    "pairwise_tie": pairwise_tie,
    "single": single,
}


def llama2_wrapper(usr_msg, sys_msg=None):
    if sys_msg is None:
        return PROMPT_INPUT_WO_SYSTEM.format(input=usr_msg)
    else:
        return PROMPT_INPUT_SYSTEM.format(input=usr_msg, system_message=sys_msg)


def build_autoj_input(prompt, resp1, resp2=None, protocol="single"):
    user_msg = protocol_mapping[protocol].format(prompt=prompt, response=resp1, response_another=resp2)
    return llama2_wrapper(user_msg, )

def extract_pairwise_result_autoj(raw_output):
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if pred_rest.startswith('response 1'): pred_label = 0
        elif pred_rest.startswith('response 2'): pred_label = 1
        elif pred_rest.startswith('tie'): pred_label = 2
    return pred_label

def extract_single_rating_autoj(score_output):
    pred_score =0.
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        pred_score = float(score_output[pos + len("Rating: [["):pos2].strip())
    else: 
        print("Warning: Rating not found in the output")
    return pred_score

