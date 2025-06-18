- split working. Works about as well as the other methods
- None of them work without supervised chains of thought
- None of them work for int max of 1000
    - not even add_normal when its given the answer!!
- AEs work beucase gradients flow from the output, decoder, latent representation, encoder, and inputs.
    - All the approaches here don't have that property becuase tokenization severs the gradients.
    - Maybe this suggests we should focus on non-tokenized, continuous thoughts? cuz differentiable?
        - Does this not matter? Am I already basically doing this through REINFORCE reparameterization trick?
    - And just to check, a single normal model which has an attn mask to the question tokens wouldn't work, right?
        - this does not capture the 

- naming conventions:
    - *add*: training a model for the toy task of modular addition
    - *think*: uses thinking tokens
        - *fixed*: instead of stopping thought rollouts once the model produces an end thought token, always sample a pre-set number of thinking tokens. runs about 20x faster.
        - *blind*: produce thinking tokens like normal. But to actually produce the next token prediction, the model can only see the thinking tokens, not the question. forces thinking tokens to contain useful information.
        - *super(vised)*: Here, instead of training for the next token prediction task (the non-thinking task) using the sampled thought tokens from the model, we use manually generated thinking tokens. Isolates training difficulties to the RL part.
            - This just makes it so that right from the start there is a 'correct' chain of thought for each question.
            - *clean*: the rl loss signal is hand crafted in some way. Probably giving 0 reward to wrong reasoning traces, positive constant to others.
        - *search*: search is used in the process of exploring the possible chains of thought during training.

- blind + super + clean works for max=100. + search also works but is not necessary.
    - The thinking policy does steadily learn the correct thinking tokens to output when given a clean loss signal only perfectly correct outputs. not so surprising.
    - So what should we loosen first?
        - my inclination is to figure out how to do rewards properly. So that would be getting rid of 'clean'

    - same combo does not work for 1000. The prediction policy can't even learn with supervised thinking sequences? bug?

    - removing clean also technically works beucase you can just softmax to recover the clean reward signal.
        - But this basically only works when the answering model knows whats going on, and probably only works in domains where perfect answers exist (so modular addition, but not general text prediction)

    - Attempts to remove 'super':
        - Hard. Both models seem to converge to a uniform distribution.
        - Tried freezing the answer model to capture some noise/structure for the thinking model to pick up on, but nothing.
        - A rephrasing of the difficulty here: we need these models to invent a language to talk to each other with.
            - At first each model knows nothing. Ignoring noise, they basically have no association between inputs and outputs.
            - If one model is speaking language, the other can learn it easily. So how to get them to invent a language the other can learn?
            - The reason an autoencoder can do  this is beucase gradients flow from output, through the decoder to intermediate result, through the encoder to the inputs.
        
- looking at addition capacity for a normal model:
    - There are 4 factors in this equation it seems:
        - input-max, d_model, n_layers, and embed/unembed tying
        - 2, 32, untied, 100  = 0.0
        - 2, 32, untied, 1000 = 6.9
        - 2, 32, tied,   100  = 0.0
        - 2, 32, tied,   1000 = 6.9
        - 2, 64, untied, 1000 = 6.9
        - 2, 64, tied,   1000 = 6.9
        
        - we can also try it while just giving it  the answer and telling it to repeat it.
        - 2, 32, untied, 1000, given answer = 2
        - 2, 32, tied,   1000, given answer = 0
        - 0, 64, tied or untied, 1000, given answer = 0
        - this suggests that the model can't even hold the 1k values properly in a 32-dim residual stream. d_model too small to ever learn to add.
    - I've tried quite large models, and I can't get them to learn 1k addition at all. how big a problem is this/do i need to change course for my proof of concept?


- This training method has several compounding levels of training difficulty
    - boostrapping problem. We have to learn to produce useful thinking tokens and learn to use thinking tokens simultaneously.
        - search
        - separate the thinking policy and the predicting policy
    - noisy/sparse rewards. The signal for the thinking token rl will be fairly noisy, even when the answer producing policy is fully trained.
        - train a value model
    - potentially large action space
        - search

- splitting the answering policy and the thinking policy into two models is kind of like a variational autoencoder.
    - read about autoencoders.

- mixture of experts? Could have some experts be solely trained for thinking and others be solely trained for answering.
    - Then the expert router can choose to think or not to think for each token.

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
                - Encouraging high *probability* variance may work, due to the fact that this requires the logprobs in question to be close to 0. Like you can have huge logprob variance but no prob variance if all the logits are very negative. But high probability variance can only be acheived if some of the logits are pretty high.
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

            - verified: If we manually create a useful chain of thought, the predicting head trivially solves the prediction task.
                - remember this is for the 'cant see the question during prediction, only the thinking tokens' version
                - I gave the model 10 thinking tokens. rollouts were made by spelling out the answer in digits.
                    - so if the answer is 52, the thought rollout it's given is length 2 and looks like [5th thinking token, 2nd thinking token, end_thought]
                - the thinking distribution does very regularly collapse it seems, so entropy term in the loss should become the default.
                - This seems like a good way to find a procedure that works for the rl component. The prediction head works, the rl now has a hill to climb. How do we get it to do that.