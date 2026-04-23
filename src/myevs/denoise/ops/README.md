本文件主要用于汇总不同算法的对比数据或与论文相关的所有需要统计的数据及结论
目前包含三个数据集：driving，ED24，LED（暂未添加）
测试指标为：AUC，F1，MESR，AOCC
所有算法实验结果都先存入csv文件中，便于复用，然后统计并绘图，并将结果路径写入README中，方便查看。
最后我需要一张汇总CSV文件的表格，包含每个算法在每个数据集上的结果，便于对比分析。


## 数据集路径
- driving: 
--轻噪：
`D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_light_slomo_shot_withlabel`
--中噪：
`D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_mid_slomo_shot_withlabel`
-- 重噪： `D:\hjx_workspace\scientific_reserach\dataset\DND21\mydriving\driving_noise_heavy_slomo_shot_withlabel/`
- ED24: 
-- myPedestrain_06:
`D:\hjx_workspace\scientific_reserach\dataset\ED24\myPedestrain_06`
1.8->轻噪；2.5->zhong噪；3.3->重噪

- LED：
暂缺

## 数据存放路径
跑完每个算法后，将结果存入csv文件中，主路径如下：
`D:\hjx_workspace\scientific_reserach\projects\myEVS\data`
子路径为数据集名称\算法名称
如ED24\myPedestrain_06\BAF\

## 噪声建模及分析
针对不同噪声环境下的噪声类型及数量进行统计汇总
针对噪声时间，空间，极性特性进行分析和总结

## 对比算法
- BAF
- STCF
- Knoise
- EvFlow
- Ynoise
- TS
- MLPF
- EBF
- EDformer（后续添加）
- Ours（n149,n175，先跑n149稳定版，n175为测试版后续还会更新）


## 消融实验
单独使用时间卷积，空间卷积，极性卷积进行测试，分析不同卷积对结果的影响