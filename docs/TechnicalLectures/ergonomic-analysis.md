# 人机工程分析

•    人机工程学是一门多学科的交叉学科，研究的核心问题是不同的作业中人、机器及环境三者间的协调，研究的目的则是通过各学科知识的应用，来指导工作器具、工作方式和工作环境的设计和改造，使得作业在效率、安全、健康、舒适等几个方面的特性得以提高。

•    人机工程学是一门新兴的边缘科学。它起源于欧洲，形成和发展于美国。

•    “人机工程学”的确切定义是，把人—机—环境系统 作为研究的基本对象，运用生理学、心理学和其它有关学科知识，根据人和机器的条件和特点，合理分配人和机器承担的操作职能，并使之相互适应，从而为人创造出舒适和安全的工作环境，使工效达到最优的一门综合性学科。

•    其实就是研究怎么让人更加舒适的使用机器

![img001](img001.png)

**人机工程的两个部分**

- 人与机器的关系

- 机器与环境的关系


•    人与机器的关系是人机工程中最重要的部分。

•    将机器进行改进，以适应人们的使用习惯。在我们的机器人身上，就分成两个方面。

•    一是机械设计部分，对加工人员、装配人员、搬运人员的友好性

•    二是程序设计部分，对操作手的友好性

**加工友好性**

•    在画图的时候要考虑是否能加工出来，毕竟我们的雕刻机能完成的功能有限，发外加工大家也都是体验过了

•    加工的方便程度也是需要考虑的地方，有些零件要经过一顿花里胡哨的操作才能加工出来，不仅费时费力，还容易失误导致报废，更惨的是发现有简易的零件可以代替，之前的努力全部白费

•    加工的重复性要尽量多，这也是标准化的一个前提，使用经过改良的简易零件，能极大的节省时间

**装配友好性**

•    装配的友好性在画图的时候体现的不是很明显，但也要重点关注。能进行装配是基础，能方便的进行装配才是重点，很多我们在图纸上觉得没问题的装配，一到了实际操作就会出各种各样的问题

**运输友好性**

•    相信大家都经历过搬运机器人，很重，也没有好下手的地方，就算几个人一起搬也是麻烦至极，所以就需要在机器人上面添加一些方便搬运的把手。

**操作手友好性**

•    根据操作手操作习惯的不同，机器人设置的反应应该也会不同

•    我之前比赛的时候，傅学长给我英雄上面放了很多功能，什么一键加速，扭腰，一键打空热量等等，但是到了场上一上头什么都忘了，键位过多对于操作手来说确实不友好，当然也有不熟悉的原因。

**机器与环境的关系**

•    在robotmaster比赛中，场地是极为重要的一个部分，做任何操作都要考虑场地因素。

•    然而在一般情况下，我们每个机器人身上的图传只能有一个，这就让操作手的视线范围极为有限，仅仅依靠机器人自身转向无法观察到四周的大部分情况，此时就需要机器人与环境的交互来为操作手提供一些帮助。

**案例**

西安交大

•    由于采用下供弹方案，步兵摩擦轮、发射机构等需要置于pitch轴线前端，造成云台发射机构重心过于靠前，如图5.1所示，此时云台重心与转轴距离为120mm，会导致其产生1.57N·m的低头力矩，会增加云台俯仰的力矩，导致pitch轴电机过热。新云台设计时将pitch轴电机置于发射机构后方，如图5.2，通过平行四连杆机构进行传动，可以保证电机轴转动角度与云台俯仰角度相同，方便电控调车，同时后移其重心，使云台重心与云台转轴距离为53.5mm，增加云台稳定性且力矩变为1.05N·m 。

•    相对于传统的下供弹方式连接复杂，安装繁琐，新的侧供弹方法简化了安装及设计，更多利用榫卯结构，且可以有效减少子弹在滑道中的运动时间，减少子弹存留量，加快供弹速度，弹丸击发延迟从下供弹的340ms减少到侧供弹的70-120ms，也优化了操作手的操作体验，提高了打击效率。

<table>
    <tr>
        <td><img src="img002.png"  />
        <td><img src="img003.png"  />
        <td><img src="img004.png"  />
        </td>
    </tr>
</table>


同济大学

<img src="img005.png" />

南方科技大学

•    案例：步兵机器人-电池仓设计。

•    问题：

•    19 年步兵机器人采用“下供弹”方案，弹舱将占据底盘巨大空间。

•    步兵底盘采用大脚车的悬架方式，电池可安放空间较小；

•    电池位置会对整车重心造成比较大的影响；

•    需求：

•    1、步兵的续航约为二十分钟，充电需约两个小时，考虑比赛换电、交通运输等因素，电池架的设计必须方便电池拆卸；

•    2、比赛中，存在跳桥落地、碰撞、弹丸攻击等而产生安全隐患， 因此池必须被良好保护；

•    3、步兵机器人在比赛中会跳桥，因此电池布置的位置需考虑整车重心；

•    4、电池位置需考虑整车布线合理；

•    解决方案：

•    电池安装从底盘下方向上，位置在整车的前部，并通过添加卡扣和魔术贴，防止电池掉落。

•    优点:

•    整车结构紧凑；

•    可以有效的平衡 miniPC，超级电容对整车造成的重心偏移；

•    卡扣+魔术贴的使用，更换电池的时候较为方便快捷，且电池不易掉落；

![img006](img006.png)

集美大学诚毅学院

![img007](img007.png)

兰州理工大学

![img008](img008.png)

郑州大学

![img009](img009.png)

![img010](img010.png)

配有超声波传感器，提高对地面高度判断的准确性和稳定性。同时利用3508电机作为驱动轮的动力系统，保证前进动力充足，四个伸缩气缸控制简单，抬升迅速，大大减少所消耗的时间。保证此方案的合理性和可操作性。

---

<p align='right'><font color=gray><strong>作者：陈新阳</strong></font></p>
---

<img src='https://img.wenhairu.com/images/2020/10/18/CbAIj.png'  >