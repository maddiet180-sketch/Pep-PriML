# Pep-PriML
Feature extraction and ML pipeline for predicting peptide–protein binding strength using signal processing–derived features.

**Significance**

Predicting the strength of peptide-protein interactions is of great interest due to the promise of peptides as flexible, tunable drug candidates. Metadynamics (MetaD) is a well-established approach to making such predictions. However, the long convergence times of MetaD, requiring multiple binding and unbinding events, limit its application in high-throughput screening. To address this limitation, we designed **Pep-PriML (Peptide Prioritization via Machine Learning)**, a machine learning workflow that utilizes fast, unconverged unbinding MetaD simulations to extract informative structural and energetic features. These features are then used to train multiple supervised regression models, reducing the dependence on fully converged trajectories. This strategy offers a more computationally efficient method of obtaining peptide-protein binding affinities, facilitating and potentially accelerating the identification and prioritization of drug candidates.
