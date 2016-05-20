# IJCAI-15 商家促销后的重复买家预测比赛纪实 

标签： ML比赛 IJCAI

---

##比赛介绍
两年一届的人工智能顶会IJCAI与阿里合办的机器学习比赛。由Alibaba inc.提供比赛数据。和去年以及今年的天池大数据竞赛赛题相似但又不同。

我们知道，国内外的电商有各种各样促（pian）销（qian）活动，像国外的黑五（Black Friday），国内的双十一等等。大量商家通过打折促销吸引顾客。而我们，aka. 剁手党，也在这一天伺机而动。其中，有许多顾客是被商家的促销所吸引的新顾客，那么他们会不会在促销之后继续在这家店买东西呢？本次比赛题目就是预测这些会重复购买的顾客。

比赛分为两个阶段（阿里比赛的一贯套路），第一阶段数据量相对较小，参赛者可以本地建模，提交最终的结果即可，最终TOP50的选手or队伍可以进入第二阶段。第二阶段则需要把代码提交到阿里的云平台上运行（当然是官方提供的接口）。

比赛的评测标准是ROC曲线下面积（AUC[^ref1]）。典型的二分类分类器评估标准。

我们队伍LinkinPark在Stage1 rank9，Stage2 rank4. 第二阶段最后几天被挤出前三。。（囧）。

##数据分析与预处理
先来看看第一阶段提供下载的数据。官方提供了两种数据格式（见下表）
|数据格式 | 文件名 | 字段|
|:-----------: | ---------:| ------:|
|data format 2|train\_format2.csv|user id,age range,gender|     
|-------------|--------------|merchant_id,label,activity log test|
|-------------|test\_format2.csv|user\_id,age\_range,gender|
|-------------|--------------|merchant_id,label,activity log test|
|data format 1|user\_log\_format1.csv|user\_id,item\_id,cat\_id,seller\_id|
|||brand\_id,time\_stamp,action\_type|
||user\_info\_format1.csv|user\_id,age\_range,gender|
||train\_format1.csv|user\_id,merchant\_id,label|
||test\_format1.csv|user\_id,merchant\_id,label|
两个数据格式都包含了双11之前6个月的用户行为日志，label表示用户是否在双11之后的6个月内重复购买，在格式2的数据中，label有-1的，表示当前用户不是当前商家的双11新用户。可见，格式一的数据是user-friendly（官方说法）的，易于做特征工程。然而，格式一中是没有label为-1的(u,m)对的，因此相比格式二会少一些信息。

我们现在将那些label为-1的数据记为Data format3，来看看format1和3的数据统计信息（下表）。
|Data format | \#Pairs | \#User |\#Merchant|\#Item | \#Cat | \#Brand|
|-----|-----:|-----:|-----:|-----:|-----:|-----:|
|data format 1 | 4215979 | 126474 | 4995 | 800322 | 1542 | 7936|
|data format 3 | 14052684 | 424170 | 4995 | 1090071 | 1658 | 8444|
然后再看看双11和非双11用户各种行为统计。
|  ----  | \#Click | \#Cart | \#Favour | \#Buy | Total|
|----|----:|----:|----:|----:|----:|
|All-Period | 48550712 | 67728 | 3277708 | 3005723 | 54901871|
|11-11 | 9188150 | 12621 | 1220285 | 156450 | 10577506|
|Daily Average Before 11-11 | 224928 | 299 | 11181 | 15485 | 240594|
双11的行为数数十数百倍于平时。。。

##特征工程
对于一般的分类算法，特征空间的选择是非常重要的，即使你的算法再牛逼，没有好的搜索空间的支持，也然并卵。因此特征工程在DM/ML比赛是大厦的基石，它决定了你的楼能盖多高。
###特征体系
统一介绍一下两个阶段用到的特征，第二阶段的特征几乎涵盖了第一阶段的。
 1. Baseline特征
    官方提供的一套为LR设计的特征体系。首先，获得每个商家的最相似的5个商家，计算方法是统计商家之间的共同用户。这些商家的id离散化作为特征（4995维稀疏特征）。然后，取当前用户对当前商家的行为特征。
 2. 信息描述类特征
    这是我们特征的主力军，时间都花在这上面了。这部分特征几乎都是具有物理意义的特征，总体可以分为三种：user，merchant，(u,m)对特征。由具体的物理意义再分为三类：
    - 计数类：行为计数、行为天计数、行为对象计数等；
    - 比值类：平均每天行为、回购率、每月行为占比等；
    - 生命周期类：用户首次/最后行为日、生存周期。
 3. 离散特征
    为什么要设计这套特征？为了使ID类完全个性化，用在LR上效果特别好（线性模型，L1正则，可实现效果好）。
    离散的方法，是将类别|ID可取的值展开成多维稀疏向量，只有对应的部分标1，其余为0.我们离散的特征为：
    - 用户的年龄和性别；
    - 当前用户在当前商家购买的商品ID、商品类别ID、品牌ID；
    - 当前用户在当前商家重复购买的品牌ID。
###特征处理
LR是一个傲娇的模型，喂的特征量纲不统一他是不会吃的- -。因此我们需要统一量纲。我们使用了log转化，即对会产生问题的特征进行了log(x)的转换。为什么不用z-score/minmax归一化方法？纯粹是因为第二阶段提供的特征提取框架很难实现。

##工具和平台
###第一阶段
scikit-learn[^sklearn]+xgboost[^xgb]，我是python党；)
xgboost是个好工具！感谢陈天奇大神带来这么牛x的GBM实现。
###第二阶段
第二阶段官方提供了一个平台“方便”参赛者提取特征和跑模型。基础平台：阿里ODPS，特征提取：Graph（管理员说的，我估计是graphlab那个框架），lr，gbrt实现：Xlib。
如果我们只用Xlib的LR和GBRT的话，我们只需要提供提取特征的代码，并打包提交即可。
这个平台让人蛋疼的几点：

 - 首先是特征提取。集群使用十台worker虚拟机，每个worker上有10个vertex，用户的日志（format2）被分别分在这些vertex上，特征提取程序分别在这些vertex上运行，达到并行的效果。这里，可以编程使每个用户的log都分配在同一个vertex上，但是每个商家就不一定了，因此如果我们在这个框架下提取商家的特征，会导致这些特征不是全局的问题- -。我们只能使用stage1的数据提取特征在作为辅助文件上传。。。
 - 其次是模型融合。官方的框架无法实现线上LR和GBDT哪怕是简单stacking learning的融合。这使得我们只能以单个模型比赛（虽然大家都一样。。）。

##模型

 - LR：Logistic Regression[^lr]，简单实用，线性模型，训练快速。可用L1/L2正则化，SGD。在面对stage2 100W+维特征时候表现很好。
 - RF：Random Forest[^rf]，各种随机，多棵树融合。stage1作为辅助模型融合。
 - GBRT：Gradient Boost Regression Tree[^gbrt]，梯度提升框架，不用特征归一，泛化强，效果赞。再次膜拜其实现Xgboost[^xgb]。
 - MFM：Multi-Instance Factorization Machines，这个模型是我们队一个大牛设计的，结合了Factorization Machines[^fm]和Multi-Intance[^mi]框架。
MFM不需要做特征工程！我们需要做的就是将每个用户的log打包，对于一个(u,m)对，一个log加上一个flag（这个log是否与m相关联）对应一个instance，一个用户的所有instance为一个bag。所以最终用户是否是商家的重复买家取决于用户所有的log是否发生在当前商家。
FM被用来预测一个instance对一个重复买家的positive贡献，由等式1定义，其中$M(x)$由等式2定义。
$$
Pr(y(x)=1;w,\{v\}) = \frac{1}{1+e^{-M(x)}}
$$
$$
M(x)=w_0+\sum_{i-1}^Dw_ix_i+\sum_{i=1}^{D}\sum_{j=i+1}^{D}<v_i,v_j>x_ix_j
$$
我们要求得目标函数是：
$$
argmax_{w,\{v\}}\prod Pr(y(B^+)=1)\prod Pr(y(B^-)=-1)
$$
使用SGD学习$w,\{v\}$，再用L-BFGS调节L2正则化后的所有权重。

##模型融合
我们第一阶段的融合框架：
![Ensemble框架](http://upload-images.jianshu.io/upload_images/196865-0adc40d61e4bcd06.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其实很简单，就是特征池中选择特征$\rightarrow$模型$\rightarrow$stacking learning[^stack]。
我们直接把MFM的结果作为一个High-level的特征加入到特征池中。提升了0.0018，0.700936$\rightarrow$0.702711.
各模型结果：
|Model | GBRT | GBRT | GBRT | RF | Final Model|
|------|------:|-----:|-----:|---:|-----------:|
|Feature Set Size | 383(all) | 276 | 254 | 383(all) | mix |
|AUC | 0.702711 | 0.702645 | 0.702585 | 0.701084 | 0.70319|

##结果
第一阶段我们队伍获得了第9。
第二阶段我们队伍获得了第4。

##总结
###历程
写一下我参加这次比赛的历程，由于今年阿里的移动推荐比赛，第一阶段就被淘汰了（T.T）才转战这个比赛的。看完赛题就着手做了起来，每天想一些新的特征，再xgb调调参数，分数一直稳步再涨，一度到了LB6。然后有几天分数就卡主上不去了，当时想着是不是应该找人组个队，分享一下思路。于是去比赛论坛上发了个贴子，当天晚上和之后的那个队长聊到了2点多钟。。。他是有老师支持的（真好），虽然当时他们名次没有我高，我还是同意让他当了队长（毕竟人家带着一帮人。。），第二天我们队又来了个大牛，就是做MFM那个，他是自己造的轮子，分数有0.689.于是我们队就在最后五天组成了。然后我们就在最后五天进行了特征融合，他们那边把特征输出给我，我来组合，跑模型，跑完的结果在给他做ensemble。
###经验&教训

 - 组队真的有必要。个人精力非常有限，组队不仅可以提供新的思路，还能使任务并行。
 - 这种比赛拼的到底是什么。我觉得1.特征工程，对数据的理解，不停地尝试都是特征工程需要做的。2.融合，之后参加了几次kaggle比赛知道了融合的强大，简单的stacking learning就能获得不少的提升。
 - 一个教训是我们在组队之初犯了许多错误，没有一个比较规范的融合模式，导致比较混乱。

##Reference
[^ref1]: Andrew P. Bradley. The use of the area under the roc curve in the evaluation of machine learning algorithms. Pattern Recognition, pages 1145–1159, 1997.

[^sklearn]: http://scikit-learn.org/stable/

[^xgb]: https://github.com/dmlc/xgboost

[^lr]: Olivier Chapelle, Eren Manavoglu, and R´omer Rosales. Simple and scalable response prediction for display advertising. ACM Transactions on Intelligent Systems and Technology, 5(4):61:1–61:34, 2014.

[^rf]: Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001.

[^gbrt]: Jerome H. Friedman. Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29:1189–1232, 1999.

[^fm]: Steffen Rendle. Factorization machines. In IEEE International Conference on Data Mining series (ICDM), pages 995–1000, 2010.

[^mi]: Oded Maron and Tom´as Lozano-P´erez. A framework for multiple-instance learning. In Advances in Neural Information Processing Systems 10, pages 570–576, 1997.

[^stack]: https://en.wikipedia.org/wiki/Ensemble_learning

