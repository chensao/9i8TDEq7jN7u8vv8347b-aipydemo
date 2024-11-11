from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler
)
from sklearn.impute import SimpleImputer

class ::DATA_PREPROCESSOR:::
    def __init__(self, scaling_method='standard'):
        """
        数据预处理器
        
        Args:
            scaling_method (str): 缩放方法
        """
        self.scaling_methods = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self.scaler = self.scaling_methods.get(scaling_method, StandardScaler())
        
    def handle_missing_values(self, dataframe, strategy='mean'):
        """
        处理缺失值
        
        Args:
            dataframe (pd.DataFrame): 输入数据
            strategy (str): 填充策略
        
        Returns:
            pd.DataFrame: 处理后的数据
        """
        imputer = SimpleImputer(strategy=strategy)
        return pd.DataFrame(
            imputer.fit_transform(dataframe), 
            columns=dataframe.columns
        )
    
    def scale_features(self, features):
        """
        特征缩放
        
        Args:
            features (np.array): 特征数组
        
        Returns:
            np.array: 缩放后的特征
        """
        return self.scaler.fit_transform(features)
