# Relational Redundancy-Free Graph Clustering


An official source code for paper "Redundancy-Free Self-Supervised Relational Learning for Graph Clustering" [Accepted by TNNLS, [[paper]](https://arxiv.org/pdf/2309.04694.pdf)] 
by Si-Yu Yi, Wei Ju, Yifang Qin, Xiao Luo, Luchen Liu, Yong-Dao Zhou, and Ming Zhang.

For questions, comments, or remarks about the code please contact Si-Yu Yi (siyuyi@mail.nankai.edu.cn). If you find this repository useful to your research or work, it is really appreciate to star this repository.


### Dependencies

The proposed R2FGC is implemented with python 3.9.7 on a NVIDIA 2204 GPU. 

Python package information is summarized as

- torch == 1.10.1
- numpy == 1.21.2
- scipy == 1.9.1
- torch_geometric == 1.7.1



### Runnings

#### To run R2FGC on ACM dataset: 
```python
python main.py --name acm --n_clusters 3 --n_input 100 --eta_value 0.2 --kappa_value 10 --epsilon_value 5e3 --lr 5e-5 --sample 256 --topk 8 --epochs 600
```

#### To run R2FGC on AMAP dataset: 
```python
python main.py --name amap --n_clusters 8 --n_input 100 --eta_value 0.2 --kappa_value 10 --epsilon_value 5e3 --lr 1e-3 --sample 256 --topk 8 --epochs 300
```

#### To run R2FGC on CITE dataset: 
```python
python main.py --name cite --n_clusters 6 --n_input 100 --eta_value 0.2 --kappa_value 10 --epsilon_value 5e3 --lr 1e-3 --sample 256 --topk 6 --epochs 600
```

#### To run R2FGC on DBLP dataset: 
```python
python main.py --name dblp --n_clusters 4 --n_input 50 --eta_value 0.2 --kappa_value 10 --epsilon_value 5e3 --lr 1e-4 --sample 256 --topk 128 --epochs 300
```

#### To run R2FGC on HHAR dataset: 
```python
python main.py --name hhar --n_clusters 6 --n_input 50 --eta_value 0.2 --kappa_value 10 --epsilon_value 5e3 --lr 1e-3 --sample 256 --topk 8 --epochs 300
```



### Citation

If you use code or datasets in this repository for your research, please cite our paper.

```
@article{yi2023redundancy,
  title={Redundancy-Free Self-Supervised Relational Learning for Graph Clustering},
  author={Yi, Si-Yu and Ju, Wei and Qin, Yifang and Luo, Xiao and Liu, Luchen and Zhou, Yong-Dao and Zhang, Ming},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```



### Acknowledgement
We would like to thank Wenxuan Tu et al. for their fascinating work (DFCN, AAAI21) and public code.

