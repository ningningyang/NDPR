# NDPR
This repository contains the source code of our paper, [Recovering Dropped Pronouns in Chinese Conversations via Modeling Their Referents](https://google.com), which is accepted for publication at [NAACL 2019](http://naacl2019.org/).

# Usage
```
python train_v4.py 
```

# Datasets
We demonstrate our model on three datasets, which are: Chinese SMS, TC section of OntoNotes Release 5.0 and BaiduZhidao. 
* Chinese SMS

This dataset is introduced in [1], which consists of 684 Chinese SMS files. In our work, we take the same train/dev/test data split as [1], which reserves 16.7% of the training set as a development set.

* OntoNotes Release 5.0(TC section)

This dataset is published in CoNLL 2012 Shared Task. We use the Chinese telephone conversation(TC) section in our work, which contains 9,507 sentences. Since the original dataset only has coreference annotations for anaphoric zero pronouns, we annotate them according to dropped pronoun recovery annotation guidelines described in [1].

* BaiduZhidao

This dataset is introduced in [2], which is a question answering dialogue dataset containing 11,160 sentences in the raw data. In our work, we make data preprocessing by splitting the entire corpus into each independent QA segmentation, removing noise data and annotating the participant information for each sentence. Our processed dataset contains 9,376 sentences.

The train/dev/test setting of these three datasets is shown in Table
<table>
    <tr>
        <td></td> 
        <td colspan="2">Train</td>
        <td colspan="2">Dev</td>
        <td colspan="2">Test</td>
   </tr>
    <tr>
        <td></td> 
        <td>Sentence</td>
        <td>DPs</td>
        <td>Sentence</td>
        <td>DPs</td>
        <td>Sentence</td>
        <td>DPs</td>
   </tr>
    <tr>
        <td>SMS</td> 
        <td>32,860</td>
        <td>25,805</td>
        <td>3,073</td>
        <td>2,395</td>
        <td>4,346</td>
        <td>3,411</td>
    </tr>
    <tr>
        <td>TC</td> 
        <td>6,562</td>
        <td>4,207</td>
        <td>1,408</td>
        <td>890</td>
        <td>1,406</td>
        <td>786</td>
    </tr>
     <tr>
        <td>BaiduZhidao</td> 
        <td>5,504</td>
        <td>4,312</td>
        <td>1,175</td>
        <td>732</td>
        <td>1,178</td>
        <td>832</td>
    </tr>
</table>


## Citation
If this work is useful in your research, please kindly cite our paper.
```
@inproceedings{yang2019text,
  title={Recovering Dropped Pronouns in Chinese Conversations via Modeling Their Referents},
  author={Jingxuan Yang, Jianzhuo Tong, Si Li, Sheng Gao, Jun Guo and Nianwen Xue},
  booktitle={NAACL},
  year={2019}
}
```

## Reference
[1] Yang, Yaqin & Liu, Yalin & Xue, Nianwen (2015). Recovering dropped pronouns from Chinese text messages. 2. 309-313. 10.3115/v1/P15-2051. 

[2] Zhang, Weinan & Liu, Ting & Yin, Qingyu , & Zhang, Yu . (2016). Neural recovery machine for chinese dropped pronoun.
