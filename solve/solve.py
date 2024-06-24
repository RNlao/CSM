from numpy import *
from math import *
import os, sympy, codecs


def main():
    path = 'test1.txt'
    try:
        file = codecs.open(path, 'r', encoding='utf-8')#path 文件路径, 'r' 只读, encoding='utf-8' 中文文件名
    except:
        print('文件不存在，请检查，程序结束运行。\n')
    data = file.readlines()#读取f文件中的每一行
    time = 0
    line = data[time].strip('\r\n')#strip()函数会根据函数体内的字符来扫描字符串从左到右删除前导和尾随的函数体内相应字符得到字符串的相应副本
    temps = line.split(' ', line.count(' '))#以空格为边界对一行的数据进行划分，即每行中形成各个元素
    num = int(temps[0])#杆件单元数
    point = int(temps[1])#节点数

    m = list(range(num)) #list形式，（0,1,2....num-1），扩充后单元刚度矩阵的列表集合
    alpha = list(range(num)) #list形式，（0,1,2....num-1），坐标转换阵
    mi =  list(range(num)) #list形式，（0,1,2....num-1），方便后面求解杆件单元末端受力，6*6大小单元刚度矩阵集合
    mm = zeros((point * 3, point * 3)) #3*节点数大小的方阵，里面元素为零（总刚度矩阵）
    uv = [0] *(point * 3) #1×（3*节点数）大小的矩阵#节点横向位移，纵向位移，转角，变为列表（list）形式
    forces = [0] * (point * 3)#1×（3*节点数）大小的矩阵，是浅拷贝#节点横向力，水平力，弯矩，列表（list）形式
    angle_1 = zeros(num)#杆件始端i角度数组
    angle_2 = zeros(num)#杆件始端j角度数组
    length = zeros(num)#长度数组
    EA = zeros(num)#截面面积与弹性模量乘积数组
    EI = zeros(num)  # 截面惯性矩与弹性模量乘积数组
    r_F0 = list(range(num)) #list形式，（0,1,2....num-1），杆件单元安装误差引起的r_F0集合
    r_F = list(range(num)) #list形式，（0,1,2....num-1），杆件单元非结点荷载引起的r_F集合
    r_Ftemp =  list(range(num)) #list形式，（0,1,2....num-1），杆件单元温差引起的r_F集合

    r_F0i = list(range(num))  # list形式，（0,1,2....num-1），杆件单元安装误差引起的r_F0集合，没有扩充
    r_Fi= list(range(num))  # list形式，（0,1,2....num-1），杆件单元非结点荷载引起的r_F集合，没有扩充
    r_Ftempi = list(range(num))  # list形式，（0,1,2....num-1），杆件单元温差引起的r_F集合，没有扩充

    deta = [0] * (num)
    R_F0 = [0] * (point * 3) #安装误差引起的总外部作用力向量（各个杆件单元的r_F0相加得到的扩充矩阵）
    R_F = [0] * (point * 3) #非结点荷载引起的总外部作用力向量（各个杆件单元的r_F相加得到的扩充矩阵）
    R_Ftemp = [0] * (point * 3) #温差作用引起的总外部作用力向量（各个杆件单元的r_F相加得到的扩充矩阵）

    if_chainpole = zeros(num)  # 判断杆件i端是否存在铰，若为1则存在
    if_i_chainpole = zeros(num)  # 铰到i端的距离
    if_j_chainpole = zeros(num)  # 铰到j端的距离
    num1 = zeros(num)#杆件始端i编号
    num2 = zeros(num)#杆件末端j编号
    F_distribute = zeros(num) #分布力
    F_concent = zeros(num) #集中力大小
    Fi_distance = zeros(num) #集中力距i结点距离
    Fj_distance = zeros(num) #集中力距j结点距离
    M_concent = zeros(num) #弯矩
    Mi_distance = zeros(num)  # 弯矩距i结点距离
    Mj_distance = zeros(num)  # 弯矩距j结点距离
    Temper1 = zeros(num) #温度t1
    Temper2 = zeros(num) #温度t2
    beita = zeros(num) #膨胀系数β
    height = zeros(num) #截面高度
    time = time + 1
    #从第二行开始读取数据
    for i in range(num):
        line = data[time].strip('\r\n')
        temps = line.split(' ', line.count(' ')) #对第二部分数据进行处理
        line4 = data[time+num+point ].strip('\r\n')
        temps4 = line4.split(' ', line4.count(' ')) #对第四部分数据进行处理，方便和杆件的节点i，j对应
        angle_1[i] = float(temps[2]) #第三个元素是角度，浮点数
        angle_2[i] = float(temps[3]) #第四个元素是角度，浮点数
        length[i] = eval(temps[4]) #第五个元素是长度，eval() 函数用来执行一个字符串表达式，并返回表达式的值。
        num1[i] = int(temps[0])#读取第i根杆件始端编号
        num2[i] = int(temps[1])#读取第i根杆件末端编号
        EA[i] = float(temps[5])
        EI[i] = float(temps[6])
        deta[i] = float(temps[7]) #杆件长度安装误差
        if_chainpole[i] = int(temps[8]) # 判断杆件是否为链杆，若为1则是链杆
        if_i_chainpole[i] = float(temps[9])#链杆到i端的距离
        if_j_chainpole[i] = float(temps[10])#链杆到j端的距离
        F_distribute[i] = float(temps4[0])  # 分布力大小
        F_concent[i] = float(temps4[1])  # 集中力大小
        Fi_distance[i] = float(temps4[2])  # 集中力距i结点距离
        Fj_distance[i] = float(temps4[3])  # 集中力距j结点距离
        M_concent[i] = float(temps4[4])  # 弯矩大小
        Mi_distance[i] = float(temps4[5])  # 弯矩距i结点距离
        Mj_distance[i] = float(temps4[6])  # 弯矩距j结点距离
        Temper1[i] = float(temps4[7])  # 温度t1大小
        Temper2[i] = float(temps4[8])  # 温度t2大小
        beita[i] = float(temps4[9])  # 膨胀系数β大小
        height[i] = float(temps4[10])  # 截面高度
        mi[i], alpha[i] = mat(angle_1[i], angle_2[i] ,length[i] , EA[i] , EI[i] ,if_chainpole[i] ,  if_i_chainpole[i] , if_j_chainpole[i] )#需声明一个生成矩阵的函数，单元矩阵函数
        # print('第{}杆的单元刚度矩阵：\n'.format(i+1),mi[i],end='\n')#输出单元刚度矩阵
        r_F0i[i] = list_Fabri_Error(angle_1[i] , angle_2[i] , length[i] , EA[i] , deta[i]) #需声明一个生成列表的函数，杆件单元的误差引起的外力
        r_Fi[i] = list_nonjointforces(F_distribute[i] , F_concent[i] , Fi_distance[i] , Fj_distance[i] , M_concent[i] , Mi_distance[i] , Mj_distance[i] , length[i] , if_chainpole[i] , if_i_chainpole[i] , if_j_chainpole[i],alpha[i])#非结点荷载对单元引起的刚度矩阵
        r_Ftempi[i] = list_temp(angle_1[i] , angle_2[i], EI[i], EA[i], Temper1[i], Temper2[i], beita[i], height[i])
        m[i] = rebig(mi[i], num1[i], num2[i], point)#需声明一个总刚度组装的函数，扩充成总刚度大小的矩阵，扩充的元素都为零
        # print(m[i],end='\n')#输出扩阶后的刚度矩阵
        r_F0[i] = list_rebig(r_F0i[i] , num1[i] , num2[i] , point)#扩充装配误差外力单元刚度矩阵
        r_F[i] = list_rebig(r_Fi[i] , num1[i] , num2[i] , point )#扩充非结点外力单元刚度矩阵
        r_Ftemp[i] = list_rebig(r_Ftempi[i] , num1[i], num2[i], point)#扩充温差作用单元刚度矩阵


        time = time + 1

    for i in range(num):
        mm = mm + m[i] #求和获得总刚度矩阵
    # print(mm,end='\n')#输出总刚度矩阵
    for i in range(num):
        R_F0 = list_sum(R_F0 , r_F0[i]) #得到总的安装误差引起的外力向量
    for i in range(num):
        R_F = list_sum(R_F , r_F[i]) #得到总的非结点荷载引起的外力向量
    for i in range(num):
        R_Ftemp = list_sum(R_Ftemp , r_Ftemp[i]) #得到温差作用引起的外力向量
    R_F_sum = list_sum(list_sum( R_F0 , R_F) ,R_Ftemp) #安装误差R_F0、非结点荷载R_F、温差作用R_Ftemp之和（扩充后的矩阵）
    
    #开始处理第三部分
    n = sympy.Symbol('n')
    for i in range(point):
        line = data[time].strip('\r\n')
        temps = line.split(' ', line.count(' '))

        uv[i * 3] = for_num(temps[0])#结点横向位移
        uv[i * 3 + 1] = for_num(temps[1])#结点纵向位移
        uv[i * 3 + 2] = for_num(temps[2])  # 结点转角
        forces[i * 3] = for_num(temps[3])#节点横向力
        forces[i * 3 + 1] = for_num(temps[4]) #结点纵向力
        forces[i * 3 + 2] = for_num(temps[5])  # 结点弯矩
        time = time + 1
    file.close()
    forces = list_sum( forces , list_nega(R_F_sum)) #取得（R-R_F_sum）
    print(forces)
    location_n = get_location_in_list(uv, n)#获得'n'在结点位移列表中的位置，'n'代表该处位移未知
    full_location = list(range(point*3)) #从0到（point*3-1）间隔为1的列表，为相减做准备
    location = diff_of_two_list(full_location,location_n) #得到非未知的位移量的位置，为求解Kuu做准备
    mms = delete(mm, [location], axis=1)#删除列
    mms = delete(mms, [location], axis=0)#删除行，得到Kuu


    if linalg.det(mms) < 0.0000000000001: #计算Kuu的行列式，若为0不可逆
        print('输入数据出错', end='\n')
        os._exit(0)
    forces_delete = delete(forces, [location], axis=0)#得到Ru

    #使用自己编写的逆矩阵函数reverse求解，缩减刚度法
    inv=reverse(mms)
    inv=((inv).dot(forces_delete.T)).T
    uv = add_add( uv, inv, location) # K的逆点乘力得到位移阵，并将已知量按照原顺序放进去，最终得到D向量

    for i in range(num):
        r_F0i[i] = list_Fabri_Error(angle_1[i], angle_2[i], length[i], EA[i], deta[i])  # 需声明一个生成列表的函数，杆件单元的误差引起的外力

        r_Fi[i] = list_nonjointforces(F_distribute[i], F_concent[i], Fi_distance[i], Fj_distance[i], M_concent[i],
                                      Mi_distance[i], Mj_distance[i], length[i], if_chainpole[i] , if_i_chainpole[i] , if_j_chainpole[i],alpha[i])  # 非结点荷载对单元引起的刚度矩阵
        r_Ftempi[i] = list_temp(angle_1[i], angle_2[i], EI[i], EA[i], Temper1[i], Temper2[i], beita[i], height[i])

    bar_force(mi, uv, r_F0i, r_Fi , r_Ftempi, num1, num2, num , point ,alpha)
    uv = matrix(uv)
    R_F = matrix(R_F)
    R_Ftemp = matrix(R_Ftemp)
    R_F0 = matrix(R_F0)
    forces = (mm.dot(uv.T)) + R_F.T +  R_Ftemp.T + R_F0.T #k*d部分加 R_F.T +  R_Ftemp.T + R_F0.T，此时为节点力

    with open('answer.txt', 'a') as f:
        f.write('各结点力为：\n')
        for i in range(point):
            f.write('结点%d力：' % (i + 1))
            f.write(str(forces[3 * i, 0]))
            f.write('  ,  ')
            f.write(str(forces[3 * i + 1, 0]))
            f.write('  ,  ')
            f.write(str(forces[3 * i+2, 0]))
            f.write('\n')
        f.write('\n各结点位移为：\n')
        for i in range(point):
            f.write('结点%d位移：(' % (i + 1))
            f.write(str(uv[0, 3 * i]))
            f.write('  ,  ')
            f.write(str(uv[0, 3 * i + 1]))
            f.write('  ,  ')
            f.write(str(uv[0, 3 * i + 2]))
            f.write(')\n')
        #输出总刚度矩阵
        f.write('\n总刚度矩阵为：\n')
        for i in range (point*3):
            for j in range (point*3):
                f.write(str('%.6f' % mm[i,j])) #保存6位小数输出)
                f.write('  ,  ')
            f.write('%d\n'%(i+1))
        #输出单元刚度矩阵
        f.write('\n单元刚度矩阵：\n')
        for i in range(num):
            f.write('杆件%d的单元刚度矩阵：\n' % (i + 1))
            for j in range(6):
                for k in range (6):
                    f.write(str('%.6f' % mi[i][j, k]))  # 保存6位小数输出)
                    f.write('  ,  ')
                f.write('%d\n' % (j + 1))
        f.close()

    print('计算完成，结果保存在"answer.txt"\n')


# 建立元素刚度矩阵（需要对其进行修改①自由度变化，②始端末端都要变化）
def mat(angle_1, angle_2, length , EA , EI , if_chainpole , if_i_chainpole , if_j_chainpole):
    CI_i = cos(angle_1 / 180 * pi)
    SI_i = sin(angle_1 / 180 * pi)
    CI_j = cos(angle_2 / 180 * pi)
    SI_j = sin(angle_2 / 180 * pi)
    alpha = matrix([[CI_i, -SI_i, 0, 0, 0, 0], [SI_i, CI_i, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, CI_j, -SI_j, 0], [0, 0, 0, SI_j, CI_j, 0], [0, 0, 0, 0, 0, 1]]) #转换阵
    k_p = matrix([[ EA/length , 0 ,0 ,-EA/length, 0,0 ] , [ 0 , 12*EI/length**3 , 6*EI/length**2 , 0 ,-12*EI/length**3 , 6*EI/length**2],
                  [0 , 6*EI/length**2 , 4*EI/length , 0 , -6*EI/length**2 , 2*EI/length ] , [-EA/length , 0 ,0 ,EA/length, 0,0],
                  [0 , -12*EI/length**3 , -6*EI/length**2 , 0 ,12*EI/length**3 , -6*EI/length**2],
                  [0 , 6*EI/length**2 , 2*EI/length , 0 , -6*EI/length**2 , 4*EI/length]]) #k'矩阵
    k = dot( dot( alpha , k_p) ,alpha.T) #全局坐标下的单元刚度矩阵
    if if_chainpole ==1: #杆件存在铰的情况
        k_p = matrix([[EA*(if_i_chainpole**3 + if_j_chainpole**3)/(3*EI*length) ,0 ,0, -EA*(if_i_chainpole**3 + if_j_chainpole**3)/(3*EI*length) ,0, 0],
                      [0, 1, if_i_chainpole ,0 ,-1 ,if_j_chainpole],
                      [0, if_i_chainpole ,if_i_chainpole**2 ,0 ,-if_i_chainpole, if_i_chainpole*if_j_chainpole],
                      [-EA*(if_i_chainpole**3 + if_j_chainpole**3)/(3*EI*length) ,0 ,0,EA*(if_i_chainpole**3 + if_j_chainpole**3)/(3*EI*length), 0, 0],
                      [0, -1, -if_i_chainpole, 0, 1, -if_j_chainpole],
                      [0, if_j_chainpole, if_i_chainpole*if_j_chainpole, 0 ,-if_j_chainpole, if_j_chainpole**2]])
        sim = float(3*EI/(if_i_chainpole**3+if_j_chainpole**3))
        for t in range(shape(k_p)[0]):
            for l in range (shape(k_p)[1]):
                k_p[t][0 , l] = sim*k_p[t][0 , l]
        k1 = dot(dot(alpha, k_p), alpha.T)  # 全局坐标下的单元刚度矩阵
        return k1, alpha
    else:
        return k, alpha


def list_Fabri_Error(angle_1, angle_2, length , EA , Fabri_Error): #针对一维的转换阵的模板，右侧的r_F，比如Fabrication error，温度作用，非结点荷载等情况
    CI_i = cos(angle_1 / 180 * pi)
    SI_i = sin(angle_1 / 180 * pi)
    CI_j = cos(angle_2 / 180 * pi)
    SI_j = sin(angle_2 / 180 * pi)
    r1 = [EA * Fabri_Error / length *CI_i, EA * Fabri_Error / length *SI_i, 0*CI_i, EA * Fabri_Error / length *(-CI_j), EA * Fabri_Error / length *(-SI_j), 0*CI_j]  #转换阵  *EA*Fabri_Error/length
    return  r1

#求解杆间力影响
def list_nonjointforces( F_distribute , F_concent , Fi_distance , Fj_distance ,M_concent , Mi_distance , Mj_distance , length ,if_chainpole , if_i_chainpole , if_j_chainpole , alpha):
    if if_chainpole==1:
        FQ = 3*F_distribute*(if_j_chainpole**4-if_i_chainpole**4)/(8*(if_j_chainpole**3+if_i_chainpole**3))
        r_F_distri = [float(0), FQ+F_distribute*if_i_chainpole, FQ*if_i_chainpole+F_distribute*if_i_chainpole**2/2, float(0),
                      -FQ+F_distribute*if_j_chainpole, FQ*if_j_chainpole-F_distribute*if_j_chainpole**2/2]
        r_F_concent = [float(0), F_concent * if_j_chainpole**3/(if_i_chainpole**3+if_j_chainpole**3),
                       F_concent * if_j_chainpole ** 3 / (if_i_chainpole ** 3 + if_j_chainpole ** 3)*if_i_chainpole,
                       float(0), F_concent * if_i_chainpole**3/(if_i_chainpole**3+if_j_chainpole**3),
                       -F_concent * if_i_chainpole ** 3 / (if_i_chainpole ** 3 + if_j_chainpole ** 3)*if_j_chainpole]
        r_F_M = [float(0), float(0), Mj_distance / length * M_concent, float(0), float(0),
                 Mi_distance / length * M_concent]
        r2 = list_sum(list_sum(r_F_distri, r_F_concent), r_F_M)
        column = array(r2)
        column=alpha.dot(column)
        r2 = [column[0,0],column[0,1],column[0,2],column[0,3],column[0,4],column[0,5]]
        return r2

    else:
        r_F_distri = [ float(0) , F_distribute*length/2 , F_distribute*(length**2)/12 , float(0) , F_distribute*length/2 , -F_distribute*(length**2)/12 ]
        r_F_concent = [ float(0) , F_concent*Fj_distance**2/length**2*(1+2*Fi_distance/length) , F_concent*Fi_distance*Fj_distance**2/length**2 ,
                        float(0) , F_concent*Fi_distance**2/length**2*(1+2*Fj_distance/length) , -F_concent*Fj_distance*Fi_distance**2/length**2  ]
        r_F_M = [ float(0) ,float(0) , Mj_distance/length*M_concent , float(0) ,float(0) , Mi_distance/length*M_concent ]
        r2 = list_sum(list_sum(r_F_distri ,  r_F_concent ) , r_F_M)
        column = array(r2)
        column=alpha.dot(column)
        r2 = [column[0,0],column[0,1],column[0,2],column[0,3],column[0,4],column[0,5]]
        return r2

#求解温度力
def list_temp (angle_1, angle_2, EI, EA, Temper1, Temper2, beita ,height):
    if Temper1==0 and Temper2==0:
        return [ float(0),float(0),float(0),float(0),float(0),float(0) ]
    else:
        CI_i = cos(angle_1 / 180 * pi)
        SI_i = sin(angle_1 / 180 * pi)
        CI_j = cos(angle_2 / 180 * pi)
        SI_j = sin(angle_2 / 180 * pi)
        return [ beita*EA*(Temper1+Temper2)/2*CI_i , beita*EA*(Temper1+Temper2)/2*SI_i , -beita*EI/height*(Temper1-Temper2) ,
                 -beita*EA*(Temper1+Temper2)/2*CI_j , -beita*EA*(Temper1+Temper2)/2*SI_j , beita*EI/height*(Temper1-Temper2)]


# 将每个单元刚度矩阵的大小，扩充到整个刚度矩阵的大小，为后面求和做准备
def rebig(m, num1, num2, point):
    if num1<num2:
        for i in range(point * 3): #刚架总刚度矩阵的维数是节点数的3倍
            if i not in [num1 * 3 - 3, num1 * 3 - 2,  num1 * 3 - 1 , num2 * 3 - 3, num2 * 3 - 2, num2 * 3 - 1]:
                m = insert(m, [i], zeros(m.shape[1]), axis=0) #insert(原数组, 插入位置, 插入值, 默认按行插入（axis=1按列插入）)，除对应节点的行外都命名为0
        for i in range(point * 3):
            if i not in [num1 * 3 - 3, num1 * 3 - 2,  num1 * 3 - 1 , num2 * 3 - 3, num2 * 3 - 2, num2 * 3 - 1]:
                m = insert(m, [i], array([zeros(m.shape[0])]).T, axis=1)#按照列插入
        return m
    else:
            m1=m.copy()
            m1[0:3,0:3]=m[3:6,3:6]
            m1[0:3,3:6]=m[3:6,0:3]
            m1[3:6,0:3]=m[0:3,3:6]
            m1[3:6,3:6]=m[0:3,0:3] 
            for i in range(point * 3): #刚架总刚度矩阵的维数是节点数的3倍
                if i not in [num1 * 3 - 3, num1 * 3 - 2,  num1 * 3 - 1 , num2 * 3 - 3, num2 * 3 - 2, num2 * 3 - 1]:
                    m1 = insert(m1, [i], zeros(m1.shape[1]), axis=0) #insert(原数组, 插入位置, 插入值, 默认按行插入（axis=1按列插入）)，除对应节点的行外都命名为0
            for i in range(point * 3):
                if i not in [num1 * 3 - 3, num1 * 3 - 2,  num1 * 3 - 1 , num2 * 3 - 3, num2 * 3 - 2, num2 * 3 - 1]:
                    m1 = insert(m1, [i], array([zeros(m1.shape[0])]).T, axis=1)#按照列插入
            return m1


def list_rebig(list, num1, num2, point): #得到扩充后的r_F阵
    lst1 = [0] * (point*3)
    lst1[int(num1*3-3):int(num1*3)]=list[0:3].copy()
    lst1[int(num2*3-3):int(num2*3)]=list[3:6].copy()
    return lst1


def diff_of_two_list( lst1 , lst2 ): #lst1—被减数，lst2—减数
    diff_list = []
    for item in lst1:
        if item not in lst2:
            diff_list.append(item)
    return diff_list


def list_sum(lst1 , lst2): #遍历求两个相同大小的列表的元素和
    sum_lst = []
    for index , item in enumerate(lst1):
        sum_lst.append(item + lst2[index])
    return sum_lst


def list_nega(lst):#让列表中的元素全部取原来的相反数
    nega = []
    for item in lst:
        nega.append(-item)
    return nega


def get_location_in_list(x, target): #target在列表x中位置，返回一个储存位置的列表
    step = -1
    items = list()
    for i in range(x.count(target)):
        y = x[step + 1:].index(target)
        step = step + y + 1
        items.append(step)
    return items


def add_add( orig_dis, add, loca): #将求解得到的位移对应着返回，orig_dis-从txt文本中读取的初始位移, add-根据矩阵位移法求得的Du, loca-orig_dis中位移已知量的位置
    t = 0
    for i in range(len(orig_dis)):
        if i not in loca:
            orig_dis[i] = add[t]
            t = t + 1
    return orig_dis



#杆件力和位移
def bar_force(mi, uv, r_F0i, r_Fi , r_Ftempi, num1, num2, num , point,  alpha ):
    with open('answer.txt', 'w') as f:
        f.write('各杆力为：\n')

        for i in range(num):
            #获得杆件对应的位移
            dis = zeros(6)

            dis[0:3]=uv[int(num1[i]* 3 - 3):int(num1[i] * 3)].copy()
            dis[3:6]=uv[int(num2[i]* 3 - 3): int(num2[i] * 3)].copy()
            r_Fi[i] =matrix(r_Fi[i])
            dis = matrix(dis)
            r_F0i[i] = matrix(r_F0i[i])
            r_Ftempi[i] = matrix(r_Ftempi[i])

            r = mi[i].dot(dis.T)+r_Fi[i].T+r_F0i[i].T+r_Ftempi[i].T
            r=alpha[i].T.dot(r)

            # print('%d%d号杆的力是'%(num1[i],num2[i]),bar_force,end='\n')
            f.write('%d%d号杆的力是' % (num1[i], num2[i]))
            for k in range(shape(r)[0]):
                f.write(str('%.6f' % r[k,0])) #保存6位小数输出
                f.write(' , ')
            f.write('\n')
        f.write('\n')
        f.close()

def bound(mm,loca_0,loca_rr):
    m_bound=mm.copy()
    for i in loca_0:
        m_bound[i,:]=0
        m_bound[:,i]=0
        m_bound[i,i]=1
    for i in loca_rr:
        m_bound[i,:]=0
        m_bound[:,i]=0
        m_bound[i,i]=1
    return m_bound


def for_num(str_):
    try:
        return float(eval(str_)) #可能有异常的代码
    except:
        return sympy.Symbol(str_)  #如果有异常就执行的代码

#使用LU分解求矩阵逆
def reverse(mms):
    n=mms.shape[1]
    #先LU分解
    L = zeros([n,n])
    U = zeros([n,n])
    for i in range(n):
        L[i][i]=1
        if i==0:
            U[0][0] = mms[0,0]
            for j in range(1,n):
                U[0][j]=mms[0,j]
                L[j][0]=mms[j,0]/U[0][0]
        else:
                for j in range(i, n):#U
                    temp=0
                    for k in range(0, i):
                        temp = temp+L[i][k] * U[k][j]
                    U[i][j]=mms[i,j]-temp
                for j in range(i+1, n):#L
                    temp = 0
                    for k in range(0, i ):
                        temp = temp + L[j][k] * U[k][i]
                    L[j][i] = (mms[j,i] - temp)/U[i][i]
    inv=zeros([n,n])
    Z=zeros([n,1]) #中间向量
    I=eye(n)
    for i in range(n):
        Z = linalg.solve(L,I[:,i])
        inv[:,i]=linalg.solve(U,Z)
    return inv



if __name__ == '__main__':
    main()

