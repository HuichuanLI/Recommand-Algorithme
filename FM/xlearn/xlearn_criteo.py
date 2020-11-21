import xlearn as xl

# 创建FFM模型
ffm_model = xl.create_ffm()
# 设置训练集和测试集
ffm_model.setTrain("./small_train.txt")
ffm_model.setValidate("./small_test.txt")

# 设置参数，任务为二分类，学习率0.2，正则项lambda: 0.002，评估指标 accuracy
param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'acc'}

# FFM训练，并输出模型
ffm_model.fit(param, './model.out')

# 设置测试集，将输出结果转换为0-1
ffm_model.setTest("./small_test.txt")
ffm_model.setSigmoid()
# 使用训练好的FFM模型进行预测，输出到output.txt
ffm_model.predict("./model.out", "./output.txt")
