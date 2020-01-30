Task List:

1) Debug code
2) Save loss values in a loss_aggregation file (this is different for reinforcement vs. imitiation)
3) Train a reinforcement model and save it (model is different for reinformcent vs. imitation)
4) Train a new model using imitation learning save the model
5) Train the model from step 5 with reinforcement and save it as a seperate model

At the end of this we will have 3 models: 
1) Only trained via reinforcement: reinforcement_{}_eps.h5 
2) Only trained via imitation: imitation_{}eps
3) Trained via imitation first then reinforcement: imitation_{}eps_rl_{}eps

*{}eps means the number of episodes on to which the training happened
