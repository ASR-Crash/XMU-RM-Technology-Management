# 有限元分析基础

**有限元分析基本原理**

有限元分析是一种分析结构受力的方法

•    最初，人们使用各种力学理论严谨地研究结构构件的受力行为。于是开发了以**“平衡”，“集合”，“本构”**三大方程为基础的分析方法。这种方法要求解**微分方程**，算出构件的位移后，再推算出应力、应变、反力等。其中“平衡”方程和“集合”方程几乎都是微分方程组

•    由于微分方程本身难以求解，这使得以上求解受力问题的效率大大降低，甚至让求解成为不可能

•    于是人们试图绕开求解微分方程。新的思路是从计算一开始去猜**结构受力后的位移**，之后算出结构在假设位移下的内外力虚功，或者是应变能，令其满足对应的能量原理。

•    整个过程没有再出现微分方程，只需要求一些积分，就可以得到位移表达式中的未知数。

•    这种新的思路带来了一个问题，若是结构位移猜的不准确，算出来的 结果与真实值的差距将非常大。

有限元是为了解决这一问题而诞生的

•    类似于微积分的思想，有限元分析将待分析的对象微分为大量的微元，然后猜测每一份上的位移，只要结构分的足够多，算出来的位移就会非常接近于真实的位移，最终降低了猜错位移对于最终结果的影响

•    综上所述，有限元分析是一种结构计算的近似方法，它利用**结构离散化**保证计算结果的准确性。

**有限元分析具体方法**

•    1.基本过程

•    2.网格绘制

•    3.应力集中与应力奇异性

**对单个零件的有限元分析**

1.根据对零件的受力分析定义夹具和外部荷载

<img src="1.png" />

2.定义网格

<img src="2.png"/>

3.运行算例

<img src="3.png" />

4.分析网格精度，精度不足需要重新定义网格（一般红色部分至少完整覆盖两层网格）

<img src="4.png"/>

**应力集中与应力奇异性**

•    应力集中：指受力构件由于外力因素或自身几何因素形状、外形尺寸发生突变而引起的局部范围内应力显著增大的现象，多出现于尖角、孔洞、缺口、沟槽以及有刚性约束处及其邻域。

•    应力奇异性：指由于受力体的几何关系，在求解应力函数的时候出现的应力无穷大问题。根据弹性理论，在尖角处的应力是无穷大，而由于**离散化误差**，有限元模型不会产生无穷大的应力结果，而是会形成**随着网格的细化，得到应力值大幅度增加的现象。**

有限元分析示例——网格定义1mm，最大应力值43210

<img src="5.png"/>

有限元分析示例——网格定义0.5mm，最大应力值54870

<img src="6.png"/>

有限元分析示例——边线网格控制0.1mm，全局网格0.8mm，最大应力值1200000

<img src="7.png" />

 有限元分析示例——加圆角模型，网格定义0.5mm

<img src="8.png"/>

---

<p align='right'><font color=gray><strong>作者：李漓江</strong></font></p>
---

<img src='https://img.wenhairu.com/images/2020/10/18/CbAIj.png'  >


