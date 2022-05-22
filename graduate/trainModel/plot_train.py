#coding:utf-8
import csv
import matplotlib
import matplotlib.pyplot as plt

exampleFile = open('train_result_10P.csv')  # 打开csv文件
exampleReader = csv.reader(exampleFile)  # 读取csv文件
exampleData = list(exampleReader)  # csv数据转换为列表
line = len(exampleData)  # 得到数据行数
# length = len(exampleData[0])  # 得到每行长度

# for i in range(1,length_zu):
#     print(exampleData[i])

step = list()
act_loss = list()
crt_loss = list()
plt.figure(figsize=(8, 8))
# ,figsize=(8, 8)
# plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
# plt.rcParams['axes.unicode_minus']=False
for i in range(1, line):  # 从第二行开始读取
    step.append(int(exampleData[i][0]))
    act_loss.append(-float(exampleData[i][1]))
    crt_loss.append(float(exampleData[i][2]))

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.subplot(211)
plt.xlim( 0,5000)
plt.ylim(0,445)
plt.xlabel("训练次数")
plt.ylabel("损失值")
l1=plt.plot(step, act_loss,color="r",label=u"行动者网络损失")  # 绘制x,y的折线图
plt.legend()
plt.subplot(212)
plt.xlim(0,5000) 
plt.ylim( 0,110) #抹掉0
plt.xlabel("训练次数")
plt.ylabel("损失值")
l2=plt.plot(step, crt_loss,color="b",label=u"评论家网络损失")  # 绘制x,y的折线图
plt.legend()

plt.savefig("./modelTrain.pdf", 
        dpi=None, 
        facecolor='w', 
        edgecolor='b',
        orientation='portrait', 
        format=None,
        transparent=False, 
        bbox_inches='tight', 
        pad_inches=0.2,
        metadata=None)

plt.show()  # 显示折线图
