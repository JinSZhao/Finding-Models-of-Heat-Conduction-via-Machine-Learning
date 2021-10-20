# Finding-Models-of-Heat-Conduction-via-Machine-Learning

## Abstract
In this paper, we develop a method for finding models of heat conduction via machine learning. Integrating machine learning and the conservation-dissipation formulism (CDF) of irreversible thermodynamics, we obtain a system of PDEs for the heat conduction. The learned PDEs satisfy the conservation-dissipation principle and thereby are hyperbolic balance laws, which can be solved by conventional numerical methods. In the training process, we use a "warm-up" technique and train a neural network by another trained neural network. Numerical tests show that the learned models can achieve very high accuracy, perform well in a long time for a wide range of Knudsen numbers, and have an excellent generalization ability.
## Numerical Experiments
- train
```python
cd train
python train_GM_Kn_Step4.py  # for G and M
python trainf_Kn.py          # for F
```

- predict
```python
cd predict
python PredictionU_Kn.py
```
- data
   - training data: data_all0428
   - predicting data: data_all0428S, data_all0428T2
   - parameters of the well-trained networks: folders in predict

## Cite
Jin Zhao, Weifeng Zhao, Zhiting Ma, Wen-An Yong, and Bin Dong. *Finding Models of Heat Conduction via Machine Learning*. Submitted and Under Review (IJHMT).
