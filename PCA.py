
import joblib

# 加载PCA模型
pca_model = joblib.load('PCA.joblib')

# 打印模型的相关信息，检查是否加载成功
print(pca_model)
