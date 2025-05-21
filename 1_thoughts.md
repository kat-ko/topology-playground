# Intersection of modularity and memory and world representations in RL

Towards Understanding the Link Between Modularity and Performance in Neural Networks for Reinforcement Learning - https://arxiv.org/pdf/2205.06451v2
- "While modularity is a common property of the best networks, balancing its function and performance can be complex and may require a specific design."
- "the MAP-Elites experiments demonstrated that the way modularity impacted network performance was dependent on other network variables like their behaviour."
- "r. Another interesting observation is that the number of neurons mapelites plot showed that the optimal modularity range decreased as the networks got bigger. This is congruent to the popular idea that modularity is a way of managing complexity in neural networks [31], [32]."
- "It may be that the lack of improvement from a modular architecture is a misleading factor, due to the complicated performance landscape of modularity."
- "Our study shows that modularity is inextricably linked with the performance of neural networks, although their relationship is nuanced."
- "The complex interdependence between modularity and other network features makes it unlikely that simple selection of modular network designs or blind optimization of modularity will be effective."
- "Three experiments demonstrated this:"
    - Observing modularity emerging from evolutionary learning, demonstrating no performance improvement from an added modularity objective
    - Mapping the conditional effect of modularity and other features on performance using MAPElites
    - Optimizing modularity while considering other network features significantly improved performance
- "Our findings suggest that modularity may be necessary for efficient network architectures but presents a challenging optimization problem."
- "This view is reinforced by the significant levels of modularity seen in the human brain and other complex systems. Limitations include generalizing findings
to backpropagation-trained networks and the need for further investigation in complex domains and the cause of entanglement."
- " Future studies could explore complex network structures such as convolutional layers [17], use larger networks, and incorporate network weights into the modularity calculations by using community detection algorithms such as [36]."




TODOs

- Analyse what makes the performance landscape of modularity so complicated


# Combination of Smarl-Worldedness and Modularity
- In Growth
    - is this something thats already encoded in growth rules?
    - does it emerge maybe, from very abstract rules?
    - Do NDPs build modules? 
- In R: Gibts sowas aerhnliches wie das paper zu modularit + rl auch fuer small-worlededness?





Generate a testbed for neural network topology performance comparison in hindsight of which nodes we define as input and output nodes. 
We wantt o compare across a watts-stroganov small-world model, a modular network with 