from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV
)

class ::RANDOM_FOREST_TRAINER:::
    def __init__(self, n_estimators=100, random_state=42):
        """
        随机森林训练器
        
        Args:
            n_estimators (int): 树的数量
            random_state (int): 随机种子
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state
        )
        
    def train_model(self, X, y):
        """
        模型训练
        
        Args:
            X (np.array): 特征
            y (np.array): 目标值
        
        Returns:
            float: 交叉验证得分
        """
        # 数据分割
        ::X_TRAIN::, ::X_TEST::, ::Y_TRAIN::, ::Y_TEST:: = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 模型训练
        self.model.fit(::X_TRAIN::, ::Y_TRAIN::)
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        return np.mean(cv_scores)
    
    def hyperparameter_tuning(self, X, y):
        """
        超参数调优
        
        Args:
            X (np.array): 特征
            y (np.array): 目标值
        
        Returns:
            dict: 最佳参数
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            estimator=self.model, 
            param_grid=param_grid, 
            cv=5
        )
        
        grid_search.fit(X, y)
        return grid_search.best_params_
