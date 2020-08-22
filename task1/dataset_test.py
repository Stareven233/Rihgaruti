import pandas as pd

path = './weibo_senti_100k.csv'
pd_all = pd.read_csv(path)

print('评论数目（总体）：%d' % pd_all.shape[0])
print('评论数目（正向）：%d' % pd_all[pd_all.label==1].shape[0])
print('评论数目（负向）：%d' % pd_all[pd_all.label==0].shape[0])

print(pd_all.sample(30))
