import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from casper import nethook
import gc
class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        load_8_bit = False,
        load_4_bit = False,
        model_state_dict = None,
        device = 'cuda:0'
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if load_8_bit:
            model = AutoModelForCausalLM.from_pretrained(model_name,load_in_8bit=True)
            nethook.set_requires_grad(False, model)
        elif load_4_bit:
            model = AutoModelForCausalLM.from_pretrained(model_name,load_in_4bit=True)
            nethook.set_requires_grad(False, model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
                )
            nethook.set_requires_grad(False, model)
            if model_state_dict is not None:
                model.load_state_dict(torch.load(model_state_dict))
                model.eval().to(device)
            else:
                    model.eval().to(device)

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )
    
def generate_outputs(current_test_cases , mt, device='cuda:0',batch_size=1, max_new_tokens=100, verbose=True):

    input_ids = mt.tokenizer(current_test_cases, padding=True, return_tensors="pt")
    input_ids['input_ids'] = input_ids['input_ids'].to(device)
    # np.save('input_ids_sec',input_ids['input_ids'].cpu().numpy())
    input_ids['attention_mask'] = input_ids['attention_mask'].to(device)
    num_input_tokens = input_ids['input_ids'].shape[1]
    # print(tokenizer.decode(input_ids['input_ids'].squeeze(0)))
    outputs = mt.model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'],
                                     max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=mt.tokenizer.pad_token_id)
    generation = mt.tokenizer.batch_decode(outputs[:, num_input_tokens:], skip_special_tokens=True)
    # print(generation)
    del outputs
    gc.collect()
    return generation


# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def predict_token(mt, prompts, return_p=False,device='cuda:0'):
    inp = make_inputs(mt.tokenizer, prompts,device=device)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p

def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    else:
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f"model.layers.{num}{'' if kind is None else '.' + kind}"