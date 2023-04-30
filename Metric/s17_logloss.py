from sklearn.metrics import log_loss

#二分类
trues = [ 1, 0, 1 ]
preds = [ 0.7, 0.1, 0.5 ]
logloss = log_loss( trues, preds )
print(logloss)
######

#多分类
trues = [ 1, 2, 0 ]
preds = [ [ 0.1, 0.9, 0.2 ], [ 0.1, 0.3, 0.9 ], [ 0.7, 0.1, 0.2 ] ]
logloss = log_loss( trues, preds )
print(logloss)
######