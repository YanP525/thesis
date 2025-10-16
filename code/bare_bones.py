from transformers import StoppingCriteria, StoppingCriteriaList

class StopAtEndThink(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.end_think_token = tokenizer.encode("</think>", add_special_tokens=False)[0]

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last token is the </think> token

        # this assumes batch size of 1
        last_token = input_ids[0][-1].item()
        if last_token == self.end_think_token:
            return True  # should stop generation, next token will be the answer
        return False  # keep generating

stop_at_end_think = StopAtEndThink(tokenizer)

with torch.no_grad():
    stopping_criteria = StoppingCriteriaList([stop_at_end_think])  # Add custom stopping criteria
    
    output = model.generate(
        inputs, 
        max_new_tokens=4096, 
        stopping_criteria=stopping_criteria
    )
    # output cotains the thinking process but NOT the final answer
    # The final answer will be the next token after the </think> token
    # lets get the end logits for the answer with a final forward pass
    end_logits = model(output, return_dict=True).logits[:, -1, :]
    # get the probabilities for the wanted answer 
    ...
