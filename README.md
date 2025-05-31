- stuff yet to try:
    - if a normal thingy works, we should try continuous thinking token version
        - might be out of scope honestly.
            - how does the model 'tell us' the token it outputted was a thinking token?
                - reserve a single residual stream element to signifiy thinking vs not?
        - would probably want to replicate coconut or something similair to learn the ropes of mixed continuous and tokenized inputs/outputs

    - take a pretrained model and just expand its embed/unembed with random stuff, then apply ther rl.
        - greatly mitigates issues with having to explore the very dense action space, hunting for the single correct next token.

    - perhaps we try training on a synthetic dataset using an even smaller model.
        - Compare a normal gpt2 style model to one with thinking tokens
        - hopefully demonstrate the thinking model taking more steps on larger additions and acheiving perfect results, where the normal model is forced to approximate

    - something other than dpo? ugh

    - When i train supervised_rollout_think while ignoring thinking token logprobs, it learns a bit (loss ~7.5), but levels out. This should be basically equivalent to nromal training just really inefficiently. Suggests a bug is present.

- OOPS: fundamental flaw:
    - how do we tell which thinking tokens were useful?
    - in both implementations (rollout_think and supervised_rollout_think):
        - we take the mean logit/logprob on correct over all rollouts in the sequence.
        - we get the logprob from the last thought token in the rollout.
        - We promote the logprobs that generated all the thinking tokens in that rollout if it was *better than the avg* accuracy for that sequence, and vice versa.
    - The problem:
        - Woll reward thinking tokens whenever the next token prediction accuracy is high.
        - Also rewards thinking tokens whenever the token is just easy to predict.
        - by my estimation this effect will be much stronger, at least at first, than the actuall lift from the thinking tokens.
    - what do to:
        - maybe you can just keep training? both signals are there, so?
            - unlikely. detect easy tokens and spew on those is much easier to learn and simpler than actually learning to use the tokens.
            - weight norm will encourage the simpler thing.
        - you could use a reference model to get a measure of prediction difficulty.
            - we use this as our baseline level of performance rather than the mean over the whole batch of rollouts.
                - reference model tells us which tokens are easy and which are hard.
            - how large/accurate does the ref model need to be?
                - speculation: the reference model need only capture relative difficulty, so a good model is not necessary
                    - counter: large models may choose to model easy+rare predictions that small models totally leave out. So the deltas may not be constant.
                - probably a paper about this. how constant are the ratios/deltas across model sizes? before/after distillation?
            - anyways even size is definitely sufficient.
        - You could do multiple rollouts given the same starting sequence, taking the mean across those as baseline.
            - I beleive this is basically the thing they did for r1, calling it GRPO.

    - It seems like training on a synthetic task is probably the best way to gain understanding and test these various ways to address the issue.

- I read quiet-star paper
    - It proposes something similair:
        - rl on intermediate hidden tokens with rewards coming from next token prediction
    - It's differences are really what I viewed as being the important part of my project. namely:
        - Still using normal language tokens as the intermedaite tokens
        - Only do it with pretrained models, fine tuned to do the reasoning?
        - Selectively start and end thinking based on the output of a start thinking token.
        - The rl loss comes from taking the same model and running it on the sequence with/without intermediate thinking to evalute which chains of thought were causally useful.
        - Methodology in general seems quite silly. making the embedding for the start thought be the same as the em-dash?
            - The unimpressive reasoning traces they show as examples support my suspicion that these design choices are not allowing the model to use the intermediate computation anywhere near optimally.
    - The choices in the paper seem strange to me, and it feels like they were aiming for something else but had to pivot to make something publishable.
    - Possibly they were trying to do what i'm doing but failed?
    - I consider this paper to not be stepping on my toes but is probably evidence that what i'm trying to do is hard to get working