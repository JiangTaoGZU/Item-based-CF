# -*- coding: utf-8 -*-
# 基于项目的协同过滤推荐算法实现
import random
import math
from operator import itemgetter
import csv
import pandas as pd

class ItemBasedCF():
    # 初始化参数
    def __init__(self):

        self.n_sim_movie = input('请输入用于推荐参考的电影数K：')
        self.n_rec_movie = input('请输入对一个用户的推荐数：')

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 初始化电影相似度矩阵
        self.movie_sim_matrix = {}
    
        self.movie_popular = {}
        self.movie_count = 0
        # 找到相似的10部电影，为目标用户推荐4部电影
        print ('='*100)
        print('一、初始并加载、划分数据......')
        print("参考电影数 =", self.n_sim_movie)
        print("推荐电影数 =", self.n_rec_movie)

    # 读文件得到“用户-电影”数据，并将数据划分为训练集与数据集
    def get_dataset(self, filename, pivot=0.75):

        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
            if(random.random() < pivot):
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = rating
                trainSet_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                testSet_len += 1
        print('划分训练集与测试集成功！')
        print('训练集长 = %s' % trainSet_len)
        print('测试集长 = %s' % testSet_len)

        # print("第一条trainSet是:",list(self.trainSet.items())[0])
        # print('\n')
        # print("第一条testSet是:",list(self.testSet.items())[0])

    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('加载 %s 文件成功！' % filename)


    # 计算电影间的相似度
    def calc_movie_sim(self):
        print ('='*100)
        print('二、计算电影的相似矩阵......')
        #建立movies_popular字典
        print ('-'*35  + '1.计算电影的流行度字典movie——popular...' +  '-'*26)
        for user, movies in self.trainSet.items():
            for movie in movies:
            #若该movie没在movies_popular字典中，则把其插入字典并赋值为0，否则+1，最终的movie_popular字典键为电影名，值为所有用户总的观看数
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                else:
                    self.movie_popular[movie] += 1
        self.movie_count = len(self.movie_popular)
        # print(self.movie_popular)
        print("训练集中电影总数 = %d" % self.movie_count)
        print ('-'*35  + '2.建立电影联系矩阵... ' + '-'*43)
        for user, movies in self.trainSet.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    #下面三步的作用是：分别将每个用户看过的每一部电影与其他所有电影的联系值置1，若之后又有用户同时看了两部电影， 则+1
                    self.movie_sim_matrix.setdefault(m1, {})
                    self.movie_sim_matrix[m1].setdefault(m2, 0)
                    self.movie_sim_matrix[m1][m2] += 1    
        print("建立电影的相似矩阵成功！")
        # print("矩阵进行相似计算前movieId=1的一行为：")
        # print(self.movie_sim_matrix['1'])  

        # 计算电影之间的相似性
        print ('-'*35  + '3.计算最终的相似矩阵...  ' + '-'*40)
        for m1, related_movies in self.movie_sim_matrix.items():
            for m2, count in related_movies.items():
                # 注意0向量的处理，即某电影的用户数为0
                if self.movie_popular[m1] == 0 or self.movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                else:
                    self.movie_sim_matrix[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
        print('计算电影的相似矩阵成功！')
        # print("电影相似矩阵中movieId=736的一行：")
        # print(self.movie_sim_matrix['736'])  



    # 针对目标用户U，找到K部相似的电影，并推荐其N部电影
    def recommend(self, user):
        K = int(self.n_sim_movie)
        N = int(self.n_rec_movie)
        rank = {}
        watched_movies = self.trainSet[user]
        for movie, rating in watched_movies.items():
            #对目标用户每一部看过的电影，从相似电影矩阵中取与这部电影关联值最大的前K部电影，若这K部电影用户之前没有看过，则把它加入rank字典中，其键为movieid名，其值（即推荐度）为w（相似电影矩阵的值）与rating（用户给出的每部电影的评分）的乘积
            for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                #计算推荐度
                rank[related_movie] += w * float(rating)
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]


    #产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print ('='*100)
        print('三、推荐并评估......')
        N = int(self.n_rec_movie)
        reuserN = input("请输入参与评估的用户数量：（总用户数为457）")
        reuserN = int(reuserN)
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()
        for user,m in list(self.trainSet.items())[:reuserN]:
            test_moives = self.testSet.get(user, {})
            rec_movies = self.recommend(user)
            print("用户 %s 的电影推荐列表为：" % user)
            self.precommend(rec_movies)
            #注意，这里的w与上面recommend的w不要一样，上面的w是指计算出的相似电影矩阵的权值，而这里是这推荐字典rank对应的推荐度
            for movie, w in rec_movies:
                if movie in test_moives:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_moives)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('准确率=%.4f\t召回率=%.4f\t覆盖率=%.4f' % (precision, recall, coverage))
        print ('='*100)

    def precommend(self,rec_m):
        csv_file = "movies.csv"
        csv_data = pd.read_csv(csv_file, low_memory = False)
        df = pd.DataFrame(csv_data)
        for movieid,w in rec_m:
            print('电影名称:',df.loc[df['movieId']==int(movieid),'title'].values,'推荐度:',w)
        
            

                
if __name__ == '__main__' :
    rating_file = "ratings.csv"
    itemCF = ItemBasedCF()
    itemCF.get_dataset(rating_file)
    itemCF.calc_movie_sim()
    itemCF.evaluate()
