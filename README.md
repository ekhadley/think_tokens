- stuff yet to try:
    - judged model rollouts version (rollout_think) should be using thinking token logprobs not thinking token logits
    - supervised rollout version (supervised_rollout_think) finished?
    - alternate approach: maybe we just do inference with gradients on?
        - does this avoid the need for rl?
        - we just sample tokens and append to the context until we get a normal token.
        - by keeping gradients during inference, when we do a backward pass on the supervised loss,
        the gradients propogate all the way through to the thinking tokens, right?

- if a normal thingy works, we should try continuous thinking token version
    - might be out of scope honestly.
        - how does the model 'tell us' the token it outputted was a thinking token?
            - reserve a single residual stream element to signifiy thinking vs not?
    - would probably want to replicate coconut or something similair to learn the ropes of mixed continuous and tokenized inputs/outputs