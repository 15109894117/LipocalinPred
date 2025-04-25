**README for LipocalinPred: A CNN-LSTM-based Method for Predicting Lipocalin Using Hybrid Features.**


**Author**

Gang Xue1,3, Quan Zou1,4, Wen Zhu1,3*, Xinyi Liao2, Shaoyou Yu1,3
1School of Mathematics and Statistics, Hainan Normal University; Hainan Haikou 571158, China.
²China Unicom (Hainan) Industrial Internet Co., Ltd, Hainan Haikou 571924, China.
3Key Laboratory of Data Science and Intelligence Education, Hainan Normal University, Ministry of Education, Hainan Haikou 571158, China.
4Yangtze Delta Region Institute (Quzhou), University of Electronic Science and Technology of China, Quzhou 324000, China
Corresponding author: Wen Zhu, E-mail: syzhuwen@163.com


**Abstract**

This paper presents a deep-learning algorithm technique based on CNN-LSTM for the identification of lipocalins, called Lipo-CNN_LSTM. Firstly, the protein sequences underwent feature extraction using three methodologies: K-mer, Composition of K-spaced Amino Acid Pairs (CKSAAP), and Correlation Coefficient Position-Specific Scoring Matrix (CC-PSSM). Then, feature integration techniques are used to fusion features, and then these features are further optimized by Principal Component Analysis (PCA) to obtain important features. In order to leverage the local feature information and the interdependencies of these features fully, we employ an effective approach that integrates the Convolutional Neural Networks (CNN) with a Long Short-Term Memory (LSTM) to detect lipocalins. An accuracy of 0.976 and 0.935 is achieved on the validation set and independent test dataset.


**Data sources**

Source: This experiment used the dataset from the open-source database UniProt.

Preprocessing :

1. we firstly collected 307 positive and 307 negative samples from the open-source database UniProt.
2. Use CD-HIT with the cutoff of 40%.
3. We exclude the sequences containing "X," "B," "Z," "J," and "O" non-amino acid characters
   
Finally, we obtain a processed data sample consisting of 209 lipocalins and 210 non-lipocalins. Furthermore, we further establish an independent dataset of 42 lipocalins and 51 non-lipocalins to validate the model.


**Code and Reproduction Guide**

Operating environment: python=3.8
1. Comparison of various individual feature representation approaches.
   Code file: Individual Data Analysis - Deep Learning.

2. Comparison of various hybrid feature representation approaches.
Code file: Combined Data Analysis (Optimal Model) - Deep Learning.

3. Performance comparison of various optimizers.
Code file: Optimizer Comparison (Optimal Data and Model) - Deep Learning.

4. Performance comparison of various classifiers.
Code file: Dimensionality Reduction Analysis (Optimal Data and Model) - Deep Learning.
