### Title:
"Un modèle adaptatif d’estimation de l’Hamiltonien par machine learning" (An Adaptive Model for Hamiltonian Estimation through Machine Learning)

### Authors:
- BELHBOUB Anouar
- BENDAOUD Hamza
- ZAOUG Imad
- BENTAHAR Saad
- ENNAJAH Ayoub

### Affiliation:
- ECOLE CENTRALE CASABLANCA, CASABLANCA, MAROC

### Abstract:
The research presents a method for inferring the dynamic parameters of a quantum system using a combination of sequential Monte Carlo and Bayesian experimental design. The algorithm is designed to efficiently balance computational and experimental resources, and it can be implemented during data collection, eliminating the need for storage and post-processing. The proposed method adapts to changes in Hamiltonian parameters and unknown noise processes between experiments. It also provides a numerical estimate of the Cramer-Rao lower bound, certifying its performance.

### Keywords:
Sequential Monte Carlo, Bayesian Experiment design, tomography, batch processing, post-processing, quadratic loss, Cramer-Rao bound, negative Bayes risk, Newton conjugate-gradient, Hamiltonien.

### Introduction:
The estimation of the Hamiltonian of a quantum system is crucial for constructing large-scale quantum information processors. Traditional methods, like quantum tomography, aim to estimate the complete process but face limitations in precision and complexity, especially for large-scale systems. The proposed method combines sequential Monte Carlo and Bayesian experimental design to efficiently estimate specific model parameters and design experiments for efficient parameter deduction. The focus is on the estimation of the Hamiltonian for large-scale quantum information processors.

### Methodology:
#### a. Bayesian Experimental Design:
   - Defines hypothesis H and parameter x for estimation.
   - Defines prior P(x) based on prior knowledge.
   - Chooses control parameters C to maximize experiment utility.
   - Collects data D based on chosen control parameters.
   - Calculates likelihood P(D|x;C) and posterior P(x|D;C) using Bayesian rules.
   - Repeats the process iteratively, updating information and control parameters.

#### b. Utility Function:
   - Defines goals of the experiment and quantifies expected gain based on different control settings.
   - Optimizes the utility function to find control settings maximizing experiment utility.

#### c. Hyperparameters:
   - Describes parameters that define the distribution of model parameters.
   - Allows for a more generalized approach in learning model parameters.

#### d. Sequential Monte Carlo:
   - Algorithm for Bayesian inference to estimate the probability distribution of an unknown parameter from observed data.
   - Involves particle propagation, weight update, and resampling.

#### e. Sequential Monte Carlo Bayesian Experimental Design (SMC-BED):
   - Optimizes experiment design using Bayesian inference and Monte Carlo methods.
   - Maximizes utility at each step, useful for situations with challenging data acquisition.

#### f. Algorithmic Part:
   - Describes the proposed model, involving particle sampling, optimization using Newton Conjugate Gradient, and iterative application.

### Results:
#### a. T2 Known:
   - Evaluates the model's performance for known T2 values.
   - Discusses the reduction in mean squared error with an increasing number of particles.

#### b. T2 Unknown:
   - Explores the model's performance when T2 follows a normal distribution.
   - Highlights the impact of the number of guesses on mean squared error.

#### c. Hyperparameter Estimation:
   - Analyzes the robustness of the algorithm for hyperparameterized processes.
   - Presents error results for hyperparameter estimation.

### Discussion and Model Improvement:
   - Introduces improvements to the model, such as a new algorithm for optimal control parameter selection.
   - Demonstrates the application of the improved model in cases with different numbers of guesses.

### Conclusion:
   - Affirms the effectiveness of the proposed algorithm (SCM-BDE) for machine learning of Hamiltonian.
   - Indicates significant improvement in Hamiltonian prediction accuracy compared to individual tools.
   - Acknowledges room for further optimization and sophistication in optimization procedures.

### References:
   - Lists relevant references used in the research.

This summary provides an overview of the article's content, methodology, results, and conclusions, capturing the key aspects of the presented research.
