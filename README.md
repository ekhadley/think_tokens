- addition task:
    - add_normal works.
    - add_think works but shows no sign of actually using the thinking tokens.
        - common failure mode:
            - first the model learns to ignore the thinking tokens and just trains like a regular model
            - with high epsilon, it the model is collapsing to putting logprob of 0 (100% prob) on a single think token and huge negative on all others.
                - epsilon exploration causes the model to randomly produce the not-constant thinking token. maybe it's learns to make proper predictions while relying on the existence of the constant thought and ignoring others. This would produce a situation where given that thinking token produces better predictions than those where we got epsilon sampled away from it? This gives us hgih logprob on token where performance is higher and (very) low logprob when pred performance is worse. hence exploding think reward. maybe?
            - When epsilon is low, and the model can't rely on being forced to produce off distn thinking tokens, it resorts to not thinking at all (even with entropy reward)

        - Another issue is how to do exploration?
            - if we use random thinking tokens, the model really is enocuraged to just ignore them.
            - Can we encourage rollout variance somehow? like a reward penalty for being far from the mean?
                - This is already built into the reward, obviously, since the rewards are mean centered.
                - And if we encourage "having better accuracy than the mean accuracy of the group" as well as "have a very different accuracy than the mean accuracy of the group" it seems like the latter is much easier to optimize for. maybe?
                - Encouraging high *probability* variance may work, due to the fact that this requires the probabilities in question to be close to 1. Like you can have huge logit variance but no prob variance if all the logits are very negative. But high probability variance can only be acheived if some of the logits are pretty high.
                    - verified by the fact that the highest prob variances ever logged were for the 10 max addition task, where the model can easily solve it without any thinking tokens.
                - does loss based on sample variance even make any sense gradient-wise? 
                    - If we give rewards based on variance of prediction probabilities, this is telling the logits for below avg rollouts to be more negative, and for above average rollouts to be more positive. We already got dat. 
                    - But the main difference is that if the logprob was already hugely negative, gradient is basically zero becuase making logprob more negative hardly changes the prob in absolute terms. But if the logprob is close to 0, this produces much larger changes in the resulting prob.

            - exploration can be done without noise via search:
                - option 1: during training, MCTS the tree of thought, with a heuristic of "if the chain of thought ended here, what would it's prediction accuracy be".
                    - Could even be an alternative to group sampling. Just take the group as the tree's endpoints.
                - option 2: train a value model to approximate what is calculated above via search. Then it could be used even at inference/when the correct answer is not known.
        
        - stab in the dark: maybe a separate model for predicting and thinking, like the target and active model used in q-learning
            - in q-learning you keep 2 versions of the model. The q-value model tells us the value of a q-state. The value of a q-state determines our policy, which we need to determine the value of a q-state, etc. We gather trajectories using the main network. We make predictions using these transitions to train the target network. Every so often, we switch the target and main networks.
            - Not sure if there's a good analogy. We need a few inputs for training:
                - thinking trajectories
                - logits of think tokens in a trajectory
                - prediction logprobs for the answer to  the addition problem
        
        - Reward the model based on attention scores to the think tokens?
            - There are other ways to learn  to ignore the thinking tokens and still have high attention. Make their embeddings or value vectors be close to 0, etc.

        - potential issue. alphazero apparently only took *one* state from each game during training, becuase correlated gradients reduce update quality.
            - we are currently training on every think token logit in a number of groups attempting to solve the same problem. giga correlated.
        
        - potentially this is a general issue. The constant part of the input is the question, and the thinking tokens are all over the place. If we train it to predict before the thinking tokens are useful, it will always just ignore them. So either we don't allow it to ignore the thinking tokens or we dont train it to predict before the thinking tokens become useful.
            - So you could have the prediction happen using *only* the thinking tokens, not the original question.
                - This works nicely for toy addition problems, but is it an issue for actual next token prediction?
                - The thoughts would have to encode literally every relevant thing for predicting the next token. So does it basically mean the thinking step does all the real work?
                    - kind of, but thats kind of the point. The new thing is that it would do all this work over multiple forward passes, recombining it all with the final singular prediction taking all the thoughts into context.

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
        - Will reward thinking tokens whenever the next token prediction accuracy is high.
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