import torch
import torch.nn as nn

from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

# utils for Yes model
prompt_finetuning="""###
{datapoint}
###
Does the previous paragraph demarcated within ### and ###
contain informative signal for fine-tuning a large-language model?
An informative datapoint should be well-formatted, contain some
usable knowledge of the world.
OPTIONS:
- Yes
- No
"""

prompt_education="""###
This is a educational datapoint to give to students during their exams:
```python
{datapoint}
###
Does the previous paragraph demarcated within ### and ###
contain informative signal to a large-language model?
An informative datapoint should be well-formatted, contain some
usable knowledge of the world.
OPTIONS:
- Yes
- No
"""
def prompt_model_id(model_id):
    if "deepseek-coder-" in model_id:
        prompt_deepseek_model= """You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science.
### Instruction:
{instruction}
### Response:
"""
        return prompt_deepseek_model
    

def build_Yes_input(datapoint , model_id ,yes_mode="finetuning"):
    prompt_model = prompt_model_id(model_id)
    if yes_mode == "finetuning":
        prompt = prompt_finetuning.format(datapoint=datapoint)
    elif yes_mode == "education":
        prompt = prompt_education.format(datapoint=datapoint)
    else:
        raise ValueError("yes_mode should be either finetuning or education")
    full_prompt = prompt_model.format(instruction=prompt)
    return full_prompt

#utils for pairrm
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




# metrics compare ranking
# from https://github.com/changyaochen/rbo/tree/master


class RankingSimilarity:
    """
    This class will include some similarity measures between two different
    ranked lists.
    """
    def __init__(
        self,
        S: Union[List, np.ndarray],
        T: Union[List, np.ndarray],
        verbose: bool = False,
    ) -> None:
        """
        Initialize the object with the required lists.
        Examples of lists:
        S = ["a", "b", "c", "d", "e"]
        T = ["b", "a", 1, "d"]

        Both lists reflect the ranking of the items of interest, for example,
        list S tells us that item "a" is ranked first, "b" is ranked second,
        etc.

        Args:
            S, T (list or numpy array): lists with alphanumeric elements. They
                could be of different lengths. Both of the them should be
                ranked, i.e., each element"s position reflects its respective
                ranking in the list. Also we will require that there is no
                duplicate element in each list.
            verbose: If True, print out intermediate results. Default to False.
        """

        assert type(S) in [list, np.ndarray]
        assert type(T) in [list, np.ndarray]

        assert len(S) == len(set(S))
        assert len(T) == len(set(T))

        self.S, self.T = S, T
        self.N_S, self.N_T = len(S), len(T)
        self.verbose = verbose
        self.p = 0.5  # just a place holder

    def assert_p(self, p: float) -> None:
        """Make sure p is between (0, 1), if so, assign it to self.p.

        Args:
            p (float): The value p.
        """
        assert 0.0 < p < 1.0, "p must be between (0, 1)"
        self.p = p

    def _bound_range(self, value: float) -> float:
        """Bounds the value to [0.0, 1.0]."""

        try:
            assert (0 <= value <= 1 or np.isclose(1, value))
            return value

        except AssertionError:
            print("Value out of [0, 1] bound, will bound it.")
            larger_than_zero = max(0.0, value)
            less_than_one = min(1.0, larger_than_zero)
            return less_than_one

    def rbo(
        self,
        k: Optional[float] = None,
        p: float = 1.0,
        ext: bool = False,
    ) -> float:
        """
        This the weighted non-conjoint measures, namely, rank-biased overlap.
        Unlike Kendall tau which is correlation based, this is intersection
        based.
        The implementation if from Eq. (4) or Eq. (7) (for p != 1) from the
        RBO paper: http://www.williamwebber.com/research/papers/wmz10_tois.pdf

        If p = 1, it returns to the un-bounded set-intersection overlap,
        according to Fagin et al.
        https://researcher.watson.ibm.com/researcher/files/us-fagin/topk.pdf

        The fig. 5 in that RBO paper can be used as test case.
        Note there the choice of p is of great importance, since it
        essentially control the "top-weightness". Simply put, to an extreme,
        a small p value will only consider first few items, whereas a larger p
        value will consider more items. See Eq. (21) for quantitative measure.

        Args:
            k: The depth of evaluation.
            p: Weight of each agreement at depth d:
                p**(d-1). When set to 1.0, there is no weight, the rbo returns
                to average overlap.
            ext: If True, we will extrapolate the rbo, as in Eq. (23).

        Returns:
            The rbo at depth k (or extrapolated beyond).
        """

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        if k is None:
            k = float("inf")
        k = min(self.N_S, self.N_T, k)

        # initialize the agreement and average overlap arrays
        A, AO = [0] * k, [0] * k
        if p == 1.0:
            weights = [1.0 for _ in range(k)]
        else:
            self.assert_p(p)
            weights = [1.0 * (1 - p) * p**d for d in range(k)]

        # using dict for O(1) look up
        S_running, T_running = {self.S[0]: True}, {self.T[0]: True}
        A[0] = 1 if self.S[0] == self.T[0] else 0
        AO[0] = weights[0] if self.S[0] == self.T[0] else 0

        for d in tqdm(range(1, k), disable=~self.verbose):

            tmp = 0
            # if the new item from S is in T already
            if self.S[d] in T_running:
                tmp += 1
            # if the new item from T is in S already
            if self.T[d] in S_running:
                tmp += 1
            # if the new items are the same, which also means the previous
            # two cases did not happen
            if self.S[d] == self.T[d]:
                tmp += 1

            # update the agreement array
            A[d] = 1.0 * ((A[d - 1] * d) + tmp) / (d + 1)

            # update the average overlap array
            if p == 1.0:
                AO[d] = ((AO[d - 1] * d) + A[d]) / (d + 1)
            else:  # weighted average
                AO[d] = AO[d - 1] + weights[d] * A[d]

            # add the new item to the running set (dict)
            S_running[self.S[d]] = True
            T_running[self.T[d]] = True

        if ext and p < 1:
            return self._bound_range(AO[-1] + A[-1] * p**k)

        return self._bound_range(AO[-1])

    def rbo_ext(self, p=0.98):
        """
        This is the ultimate implementation of the rbo, namely, the
        extrapolated version. The corresponding formula is Eq. (32) in the rbo
        paper.
        """

        self.assert_p(p)

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        # since we are dealing with un-even lists, we need to figure out the
        # long (L) and short (S) list first. The name S might be confusing
        # but in this function, S refers to short list, L refers to long list
        if len(self.S) > len(self.T):
            L, S = self.S, self.T
        else:
            S, L = self.S, self.T

        s, l = len(S), len(L)  # noqa

        # initialize the overlap and rbo arrays
        # the agreement can be simply calculated from the overlap
        X, A, rbo = [0] * l, [0] * l, [0] * l

        # first item
        S_running, L_running = {S[0]}, {L[0]}  # for O(1) look up
        X[0] = 1 if S[0] == L[0] else 0
        A[0] = X[0]
        rbo[0] = 1.0 * (1 - p) * A[0]

        # start the calculation
        disjoint = 0
        ext_term = A[0] * p

        for d in tqdm(range(1, l), disable=~self.verbose):
            if d < s:  # still overlapping in length

                S_running.add(S[d])
                L_running.add(L[d])

                # again I will revoke the DP-like step
                overlap_incr = 0  # overlap increment at step d

                # if the new items are the same
                if S[d] == L[d]:
                    overlap_incr += 1
                else:
                    # if the new item from S is in L already
                    if S[d] in L_running:
                        overlap_incr += 1
                    # if the new item from L is in S already
                    if L[d] in S_running:
                        overlap_incr += 1

                X[d] = X[d - 1] + overlap_incr
                # Eq. (28) that handles the tie. len() is O(1)
                A[d] = 2.0 * X[d] / (len(S_running) + len(L_running))
                rbo[d] = rbo[d - 1] + 1.0 * (1 - p) * (p**d) * A[d]

                ext_term = 1.0 * A[d] * p**(d + 1)  # the extrapolate term

            else:  # the short list has fallen off the cliff
                L_running.add(L[d])  # we still have the long list

                # now there is one case
                overlap_incr = 1 if L[d] in S_running else 0

                X[d] = X[d - 1] + overlap_incr
                A[d] = 1.0 * X[d] / (d + 1)
                rbo[d] = rbo[d - 1] + 1.0 * (1 - p) * (p**d) * A[d]

                X_s = X[s - 1]  # this the last common overlap
                # second term in first parenthesis of Eq. (32)
                disjoint += 1.0 * (1 - p) * (p**d) * \
                    (X_s * (d + 1 - s) / (d + 1) / s)
                ext_term = 1.0 * ((X[d] - X_s) / (d + 1) + X[s - 1] / s) * \
                    p**(d + 1)  # last term in Eq. (32)

        return self._bound_range(rbo[-1] + disjoint + ext_term)
