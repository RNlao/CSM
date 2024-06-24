from numpy import *
from math import *
import sys,os,sympy,codecs
 
def main():
    path=input('请输入文件名：')
    try:
        file=codecs.open(path,'r',encoding='utf-8')
    except:
        print('文件不存在，请检查，程序结束运行。\n')
    data=file.readlines()
    time=0
    line=data[time].strip('\r\n')
    temps=line.split(' ',line.count(' '))
    num=int(temps[0]) #杆件数
    point=int(temps[1]) #节点数
    #各种元素初始化
    m=list(range(num))
    mi=list(range(num))
    mm=zeros((point*3,point*3))
    uv=[0]*(point*3)#位移和转角
    forces=[0]*(point*3)#杆端力
    angle_i=zeros(num)#杆端夹角
    angle_j=zeros(num)
    length=zeros(num)#杆长度
    EA=zeros(num)
    EI=zeros(num)
    num1=zeros(num)
    num2=zeros(num)
    r_F0=list(range(num))#安装误差引起的力
    r_F=list(range(num))#非节点荷载引起
    r_Ftemp=list(range(num))#温度引起
 
    time=time+1
 
    for i in range(num):
        line=data[time].strip('\r\n')
        temps=line.split(' ',line.count(' '))
        angle[i]=float(temps[2])
        length[i]=eval(temps[3])
        num1[i]=int(temps[0])
        num2[i]=int(temps[1])
        m[i]=mat(angle[i],length[i])
        # print(m[i],end='\n')#输出刚度矩阵
        m[i]=rebig(m[i],num1[i],num2[i],point)
        # print(m[i],end='\n')#输出扩阶后的刚度矩阵
        time=time+1
 
    for i in range(num):
        mm=mm+m[i]
    #print(mm,end='\n')#输出总刚度矩阵
    for i in range(point):
        line=data[time].strip('\r\n')
        temps=line.split(' ',line.count(' '))
 
 
        uv[i*2]=float(temps[0])
        uv[i*2+1]=float(temps[1])
        forces[i*2]=for_num(temps[2])
        forces[i*2+1]=for_num(temps[3])
        time=time+1
    file.close()        
    location=get_location_in_list(uv.tolist(),0)
    mms=delete(mm,[location], axis=1) 
    mms=delete(mms,[location], axis=0)
    if linalg.det(mms)<0.00000000001:
        print('输入数据出错',end='\n')
        os._exit(0)
    forces_delete=delete(forces,[location], axis=0)
 
    uv=add_add(point,uv,((mms.I).dot(forces_delete.T)).T,location)
    bar_force(angle,uv,length,num1,num2,num)
    uv=matrix(uv)
    forces=(mm.dot(uv.T))
    
    with open('计算结果.txt','a') as f:
        f.write('各结点力为：\n')
        for i in range(point):
            f.write('结点%d力：'%(i+1))
            f.write(str(forces[2*i,0]))
            f.write('  ,  ')
            f.write(str(forces[2*i+1,0]))
            f.write('\n')
        f.write('\n各结点位移为：\n')
        for i in range(point):
            f.write('结点%d位移：L/EA('%(i+1))
            f.write(str(uv[0,2*i],))
            f.write('  ,  ')
            f.write(str(uv[0,2*i+1]))
            f.write(')\n')
        f.close()
    #for i in range(point):
        #print('结点%d力：'%(i+1),forces[2*i,0],',',forces[2*i+1,0],end='\n')
    #for i in range(point):
        #print('结点%d位移：L/EA('%(i+1),uv[0,2*i],',',uv[0,2*i+1],end=')\n')
 
    print('计算完成，结果保存在"计算结果.txt"\n')
 
 
#建立元素刚度矩阵
def mat(angle,length):
    lamda=cos(angle/180*pi)
    miu=sin(angle/180*pi)
    return (matrix([[lamda**2,miu*lamda,-lamda**2,-lamda*miu],
                         [miu*lamda,miu**2,-lamda*miu,-miu**2],
                         [-lamda**2,-miu*lamda,lamda**2,lamda*miu],
                         [-lamda*miu,-miu**2,lamda*miu,miu**2]]))/length
 
 
#组装总刚度矩阵
def rebig(m,num1,num2,point):
    for i in range(point*2):
        if i not in [num1*2-2,num1*2-1,num2*2-2,num2*2-1]:
            m=insert(m,[i],zeros(m.shape[1]),axis=0)
    for i in range(point*2):
        if i not in [num1*2-2,num1*2-1,num2*2-2,num2*2-1]:      
            m=insert(m,[i],array([zeros(m.shape[0])]).T,axis=1)
    return m
 
def get_location_in_list(x, target):
    step = -1
    items = list()
    for i in range(x.count(target)):
        y = x[step + 1:].index(target)
        step = step + y + 1
        items.append(step)
    return items
 
def add_add(point,dis,add,loca):
    temp=[0]*(point*2)
    t=0
    for i in range(point*2):
         if i not in loca:
             temp[i]=add[t,0]
             t=t+1
         else:
             temp[i]=0
    return temp
 
def bar_force(angle,uv,length,num1,num2,num):
    with open('计算结果.txt','w') as f:
        f.write('各杆力为：\n')
        for i in range(num):
            lamda=cos(angle[i]/180*pi)
            miu=sin(angle[i]/180*pi)      
            bar_force=(-lamda*uv[int(num1[i]*2-2)]-miu*uv[int(num1[i]*2-1)]+lamda*uv[int(num2[i]*2-2)]+miu*uv[int(num2[i]*2-1)])/length[i]
            #print('%d%d号杆的力是'%(num1[i],num2[i]),bar_force,end='\n')
            f.write('%d%d号杆的力是'%(num1[i],num2[i]))
            f.write(str(bar_force))
            f.write('\n')
        f.write('\n')
        f.close()
 
def for_num(str_):
    try:
        return float(eval(str_))
    except:
        return sympy.Symbol(str_)
 
if __name__=='__main__':
    main()
