# TransFG: A Transformer Architecture for Fine-Grained Recognition

## Introduction

In the paper, we present the first study which explores the potential of vision transformers in the context of fine-grained visual classification. We find that directly applying ViT on FGVC already produces satisfactory results while a lot of adaptations according to the characteristics of FGVC can be applied to further boost the performance. To be specific, we propose Part Selection Module which can find the discriminative regions and remove redundant information. A contrastive loss is introduced to make the model more discriminative. We name this novel yet simple transformer-based framework TransFG, and evaluate it extensively on five popular fine-grained visual classification benchmarks (CUB-200-2011, Stanford Cars, Stanford Dogs, NABirds, iNat2017).

## Method

We first briefly review the framework of vision transformer and show how to do some preprocessing steps to extend it into fine-grained recognition. Then, the overall framework of TransFG will be elaborated.

### Vision transformer as feature extractor

**Image Sequentialization.** Following ViT, we first preprocess the input image into a sequence of flattened patches $x_p$. However, the original split method cut the images into non-overlapping patches, which harms the local neighboring structures especially when discriminative regions are split. To alleviate this problem, we propose to generate overlapping patches with sliding window. To be specific, we denote the input image with resolution $H * W$, the size of image patch as $P$ and the step size of sliding window as $S$. Thus the input images will be split into N patches where

$$
\tag{equ:split}
    N = N_H * N_W = \lfloor \frac{H - P + S}{S} \rfloor * \lfloor \frac{W - P + S}{S} \rfloor
$$

In this way, two adjacent patches share an overlapping area of size $(P - S) * P$ which helps to preserve better local region information. Typically speaking, the smaller the step $S$ is, the better the performance will be. But decreasing S will at the same time requires more computational cost, so a trade-off needs to be made here.

**Patch Embedding.** We map the vectorized patches $x_p$ into a latent D-dimensional embedding space using a trainable linear projection. A learnable position embedding is added to the patch embeddings to retain positional information as follows:

$$
    \mathbf{z}_0 = [x_p^1\mathbf{E},x_p^2\mathbf{E},\cdots,x_p^N\mathbf{E}] + \mathbf{E}_{pos}
$$

where $N$ is the number of image patches, $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) * D}$ is the patch embedding projection, and $\mathbf{E}_{pos} \in \mathbb{R}^{N * D}$ denotes the position embedding.

The Transformer encoder contains $L$ layers of multi-head self-attention (MSA) and multi-layer perceptron (MLP) blocks. Thus the output of the $l$-th layer can be written as follows:

$$
    \mathbf{z}^{'}_{l} = MSA(LN(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1} l \in 1, 2, \cdots, L \\
    
    \mathbf{z}_{l} = MLP(LN(\mathbf{z}^{'}_{l})) + \mathbf{z}^{'}_{l} l \in 1, 2, \cdots, L 
$$

where $LN(\cdot)$ denotes the layer normalization operation and $\mathbf{z}_l$ is the encoded image representation. ViT exploits the first token of the last encoder layer $z_{L}^{0}$ as the representation of the global feature and forward it to a classifier head to obtain the final classification results without considering the potential information stored in the rest of the tokens.

### TransFG Architecture

While our experiments show that the pure Vision Transformer can be directly applied into fine-grained visual classification and achieve impressive results, it does not well capture the local information required for FGVC. To this end, we propose the Part Selection Module (PSM) and apply contrastive feature learning to enlarge the distance of representations between confusing sub-categories.

#### Part Selection Module

One of the most important problems in fine-grained visual classification is to accurately locate the discriminative regions that account for subtle differences between similar sub-categories. For example, Fig 3 shows a confusing pair of images from the CUB-200-2011 (citation) dataset. The model needs to have the ability to capture the very small differences, i.e., the color of eyes and throat in order to distinguish these two bird species. Region proposal networks and weakly-supervised segmentation strategies are widely introduced to tackle this problem in the traditional CNN-based methods.

Vision Transformer model is perfectly suited here with its innate multi-head attention mechanism. To fully exploit the attention information, we change the input to the last Transformer Layer. Suppose the model has K self-attention heads and the hidden features input to the last layer are denoted as $\mathbf{z}_{L-1}=[z_{L-1}^{0};z_{L-1}^{1},z_{L-1}^{2},\cdots,z_{L-1}^{N}]$. The attention weights of the previous layers can be written as follows:

$$
    \mathbf{a}_l=[a_{l}^{0},a_{l}^{1},a_{l}^{2},\cdots,a_{l}^{K}]  l \in 1, 2, \cdots, L - 1 \\
     a_{l}^{i} = [a_{l}^{i_{0}};a_{l}^{i_{1}},a_{l}^{i_{2}},\cdots,a_{l}^{i_{N}}]  i \in 0, 1, \cdots, K - 1
$$

Previous works suggested that the raw attention weights do not necessarily correspond to the relative importance of input tokens especially for higher layers of a model, due to lack of token identifiability of the embeddings. To this end, we propose to integrate attention weights of all previous layers. To be specific, we recursively apply a matrix multiplication to the raw attention weights in all the layers as

$$
    \mathbf{a}_{final} = \prod_{l=0}^{L-1}\mathbf{a}_l
$$

As $\mathbf{a}_{final}$ captures how information propagates from the input layer to the embeddings in higher layers, it serves as a better choice for selecting discriminative regions compared to the single layer raw attention weights ${a}_{L-1}$. We then choose the index of the maximum value $A_1, A_2, \cdots, A_K$ with respect to the K different attention heads in $\mathbf{a}_{final}$. These positions are used as index for our model to extract the corresponding tokens in $\mathbf{z}_{L-1}$. Finally, we concatenate the selected tokens along with the classification token as the input sequence which is denoted as:

$$
    \mathbf{z}_{local} = [z_{L-1}^{0};z_{L-1}^{A_{1}},z_{L-1}^{A_{2}},\cdots,z_{L-1}^{A_{K}}]
$$

By replacing the original entire input sequence with tokens corresponding to informative regions and concatenate the classification token as input to the last Transformer Layer, we not only keep the global information but also force the last Transformer Layer to focus on the subtle differences between different sub-categories while abandoning less discriminative regions such as background or common features among a super class.

#### Contrastive feature learning

Following ViT, we still adopt the first token $z_i$ of the PSM module for classification. A simple cross-entropy loss is not enough to fully supervise the learning of features since the differences between sub-categories might be very small. To this end, we adopt contrastive loss $\mathcal{L}_{con}$ which minimizes the similarity of classification tokens corresponding to different labels and maximizes the similarity of classification tokens of samples with the same label $y$. To prevent the loss being dominated by easy negatives (different class samples with little similarity), a constant margin $\alpha$ is introduced that only negative pairs with similarity larger than $\alpha$ contribute to the loss $\mathcal{L}_{con}$. Formally, the contrastive loss over a batch of size $B$ is denoted as:

$$
\tag{equ:con}
    
    \mathcal{L}_{con} = \frac{1}{B^2}\sum_i^B[\sum_{j:y_i=y_j}^{B}(1-Sim(z_i,z_j)+ \\ \sum_{j:y_i \neq y_j}^{B}\max((Sim(z_i,z_j)-\alpha), 0)]

$$

where $z_i$ and $z_j$ are pre-processed with $l2$ normalization and $Sim(z_i,z_j)$ is thus the dot product of $z_i$ and $z_j$.

In summary, our model is trained with the sum of cross-entropy loss $L_{cross}$ and contrastive $L_{con}$ together which can be expressed as:

$$
    \mathcal{L} = \mathcal{L}_{cross}(y, y') + \mathcal{L}_{con}(z)
$$

where $\mathcal{L}_{cross}(y,y')$ is the cross-entropy loss between the predicted label $y'$ and the ground-truth label $y$.

## Experiments

In this section, we first introduce the detailed setup including datasets and training hyper-parameters. Quantitative analysis is then given followed by ablation studies. We further give qualitative analysis and visualization results to show the interpretability of our model.

### Experiments Setup

**Datasets.** We evaluate our proposed TransFG on five widely used fine-grained benchmarks, i.e., CUB-200-2011 , Stanford Cars , Stanford Dogs , NABirds  and iNat2017 . We also exploit its usage in large-scale challenging fine-grained competitions.

**Implementation details.**

Unless stated otherwise, we implement TransFG as follows. First, we resize input images to $448 * 448$ except $304 * 304$ on iNat2017 for fair comparison (random cropping for training and center cropping for testing). We split image to patches of size 16 and the step size of sliding window is set to be 12. Thus the $H, W, P, S$ in Eq \ref{equ:split} are 448, 448, 16, 12 respectively. The margin $\alpha$ in Eq \ref{equ:con} is set to be 0.4. We load intermediate weights from official ViT-B\_16 model pretrained on ImageNet21k. The batch size is set to 16. SGD optimizer is employed with a momentum of 0.9. The learning rate is initialized as 0.03 except 0.003 for Stanford Dogs dataset and 0.01 for iNat2017 dataset. We adopt cosine annealing as the scheduler of optimizer.

All the experiments are performed with four Nvidia Tesla V100 GPUs using the PyTorch toolbox and APEX.

### Ablation Study

We conduct ablation studies on our TransFG pipeline to analyze how its variants affect the fine-grained visual classification result. All ablation studies are done on CUB-200-2011 dataset while the same phenomenon can be observed on other datasets as well.

**Influence of image patch split method.** We investigate the influence of our overlapping patch split method through experiments with standard non-overlapping patch split. Both on the pure Vision Transformer and our improved TransFG framework, the overlapping split method bring consistently improvement, i.e., 0.2\% for both frameworks. The additional computational cost introduced by this is also affordable as shown in the fourth column. 

**Influence of Part Selection Module.** By applying the Part Selection Module (PSM) to select discriminative part tokens as the input for the last Transformer layer, the performance of the model improves from 90.3\% to 91.0\%. We argue that this is because in this way, we sample the most discriminative tokens as input which explicitly throws away some useless tokens and force the network to learn from the important parts.

**Influence of contrastive loss.** We observe that with contrastive loss, the model obtains a big performance gain. Quantitatively, it increases the accuracy from 90.3\% to 90.7\% for ViT and 91.0\% to 91.5\% for TransFG. We argue that this is because contrastive loss can effectively enlarge the distance of representations between similar sub-categories and decrease that between the same categories.

**Influence of margin $\alpha$.** We find that a small value of the margin $\alpha$ will lead the training signals dominated by easy negatives thus decrease the performance while a high value of $\alpha$ hinder the model to learn sufficient information for increasing the distances of hard negatives. Empirically, we find 0.4 to be the best value of $\alpha$ in our experiments.