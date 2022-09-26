# RMS-HRec
This is the source code of RMS-HRec. 

__Paper__: Automatic Meta-Path Discovery for Effective Graph-Based Recommendation (CIKM'22)

### Environment
```
pytorch == 1.5.0
dgl == 0.6.0
```
Note that `dgl` should be installed according to the instructions on [DGL official website](https://www.dgl.ai/pages/start.html)

### To run the code
```shell
python train_fm.py --data_name yelp_data
```


If you use our datasets or codes, please cite our paper.
```
@inproceedings{RMS-HRec,
  author    = {Wentao Ning and
               Reynold Cheng and
               Jiajun Shen and
               Nur Al Hasan Haldar and
               Ben Kao and
               Xiao Yan and
               Nan Huo and
               Wai Kit Lam and
               Tian Li and
               Bo Tang},
  title     = {Automatic Meta-Path Discovery for Effective Graph-Based Recommendation},
  booktitle = {CIKM},
  year      = {2022}
}
```
