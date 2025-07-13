
- naming conventions:
    - *add*: training a model for the toy task of modular addition
    - *think*: uses thinking tokens
        - *fixed*: instead of stopping thought rollouts once the model produces an end thought token, always sample a pre-set number of thinking tokens. runs about 20x faster.
        - *blind*: produce thinking tokens like normal. But to actually produce the next token prediction, the model can only see the thinking tokens, not the question. forces thinking tokens to contain useful information.
        - *super(vised)*: Here, instead of training for the next token prediction task (the non-thinking task) using the sampled thought tokens from the model, we use manually generated thinking tokens. Isolates training difficulties to the RL part.
            - This just makes it so that right from the start there is a 'correct' chain of thought for each question.
                - This avoids the need to 'invent a language' to communicate to the answering policy. Under 'super' variant, a language exists and the thinking policy just needs to learn it.
            - *clean*: the rl loss signal is hand crafted in some way. Probably giving 0 reward to wrong reasoning traces, positive constant to others.
        - *search*: search is used in the process of exploring the possible chains of thought during training, as opposed to epsilon-greedy exploration or no exploration.

- blind + super + clean works for max=100. + search also works but is not necessary.
    - The thinking policy does steadily learn the correct thinking tokens to output when given a clean loss signal only perfectly correct outputs. not so surprising.
    - So what should we loosen first?
        - my inclination is to figure out how to do rewards properly. So that would be getting rid of 'clean'

    - same combo does not work for 1000. The prediction policy can't even learn with supervised thinking sequences? bug?
        - Not bug, just had bad hyperparams apparently. need large batch sizes >= 64 and weight decay is evil.

    - removing clean also technically works beucase you can just softmax to recover the clean reward signal.
        - But this basically only works when the answering model knows whats going on, and probably only works in domains where perfect answers exist (so modular addition, but not general text prediction)
            - Although final answer logits in normal LMs are quite sparse/spiky. So not a single correct answer like arithmetic tasks, but close to one hot. few-hot. Could still be applicable on normal language.

    - Attempts to remove 'super':
        - Hard. Both models seem to converge to a uniform distribution.
        - Tried freezing the answer model to capture some noise/structure for the thinking model to pick up on, but nothing.
        - A rephrasing of the difficulty here: we need these models to invent a language to talk to each other with.
            - At first each model knows nothing. Ignoring noise, they basically have no association between inputs and outputs.
            - If one model is speaking language, the other can learn it easily. So how to get them to invent a language the other can learn?
            - The reason an autoencoder can do  this is beucase gradients flow from output, through the decoder to intermediate result, through the encoder to the inputs.

- This training method has several compounding levels of training difficulty
    - boostrapping problem. We have to learn to produce useful thinking tokens and learn to use thinking tokens simultaneously.
        - search
        - separate the thinking policy and the predicting policy
    - noisy/sparse rewards. The signal for the thinking token rl will be fairly noisy, even when the answer producing policy is fully trained.
        - train a value model
    - potentially large action space
        - search

- is modular addition just a bad task? I think so...
    - The thing I want to get at is algorithmic complexity.
        - As in normal transformers are limited in the complexity (# steps) of the algorithms they can learn to apply
        - thinking tokens allow unsupervised splitting over arbitrary length algorithms.
    - But is modular addition even complexity bounded? It seems like no?
        - higher  additions werent working until i tweaked hyperparams and raised d_model.
        - Seems more like bandwidth, information bottleneck squeeze limited.
        - I guess that's one of the key surprises of transformers: wide algorithms (memorization, heuristics, shortcuts) get you quite far!
        - You need to actually differentiate between naive complexity (how many operations are needed), and serial complexity.
            - Modular addition, as the grokking paper teaches us, is highly parallelizable, making the input size limit a matter of resid stream info capacity.
    - So perhaps a different task?
        - graph search task? encode graphs as lists of nodes or something?
        - Multiple modular additions in a single problem?
        - simple sat problems?
        - list sorting?
        - maybe no more toy problems and just do the thing?
    - But idk, isnt the whole point that it might learn an alternative strategy? 
        - Like if it's in the bandwidth bottlenecked regime on addition and has the opportunity to split its computation among multiple steps and it can't learn to do that, then the thing just doesn't work.

- AEs work beucase gradients flow from the output, decoder, latent representation, encoder, and inputs.
    - All the approaches here don't have that property becuase tokenization severs the gradients.
    - Maybe this suggests we should focus on non-tokenized, continuous thoughts? cuz differentiable?
        - Does this not matter? Am I already basically doing this through REINFORCE reparameterization trick?
    - And just to check, a single normal model which has an attn mask to the question tokens wouldn't work, right?
        - ?