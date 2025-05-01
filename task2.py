import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score
import time
import warnings
import os

warnings.filterwarnings('ignore')

# 这是要安全读入，避免格式不兼容
def safe_read_csv(file_path, encoding='latin1'):
    """安全读取CSV文件，处理可能的异常"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        raise


# 1) 数据读取与标签处理
def load_and_preprocess_data():
    # 读取四个CSV文件并添加标签
    data_dir = 'dataSetCsv/'
    try:
        adware = safe_read_csv(os.path.join(data_dir, 'adware.csv'))
        adware['label'] = 'adware'

        normal = safe_read_csv(os.path.join(data_dir, 'normal.csv'))
        normal['label'] = 'normal'

        trojan = safe_read_csv(os.path.join(data_dir, 'trojan.csv'))
        trojan['label'] = 'trojan'

        viaxmr = safe_read_csv(os.path.join(data_dir, 'viaxmr.csv'))
        viaxmr['label'] = 'viaxmr'
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise

    # 合并数据集
    df = pd.concat([adware, normal, trojan, viaxmr], axis=0, ignore_index=True)
    print(f"初始数据量: {len(df)}条记录")

    # 删除字符类型特征（更安全的方法）
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        raise ValueError("没有找到数值型特征列")

    # 确保label列存在
    if 'label' not in df.columns:
        raise ValueError("数据中缺少label列")

    df = df[numeric_cols.append(pd.Index(['label']))]
    print(f"选择数值特征后数据量: {len(df)}条记录")

    # 数据清洗 - 更健壮的方法
    # 1. 删除完全重复的行
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"删除{initial_count - len(df)}条完全重复的记录")

    # 2. 处理缺失值 - 先删除全为NA的列
    df = df.dropna(axis=1, how='all')
    # 然后删除包含太多NA的行（阈值设为50%）
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)
    # 最后用中位数填充剩余的NA
    for col in numeric_cols:
        if col in df.columns:  # 确保列仍然存在
            df[col] = df[col].fillna(df[col].median())

    print(f"处理缺失值后数据量: {len(df)}条记录")

    # 3. 异常值处理 - 更安全的方法
    for col in numeric_cols:
        if col in df.columns and df[col].nunique() > 1:  # 只在有变化的情况下处理
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:  # 避免除零错误
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"处理异常值后数据量: {len(df)}条记录")

    # 检查是否还有数据
    if len(df) == 0:
        raise ValueError("数据清洗后没有剩余数据")

    # 归一化处理（不包括标签列）
    try:
        scaler = MinMaxScaler()
        features = df.drop('label', axis=1)
        df[features.columns] = scaler.fit_transform(features)
    except Exception as e:
        print(f"归一化时出错: {str(e)}")
        raise

    # 将标签编码为数字
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    return df, le


# 2) 数据集划分
def split_data(df):
    """划分训练集和测试集"""
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\n数据集划分结果:")
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# 3) 模型训练与测试
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """训练和评估多个模型"""
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42),
        'Naive Bayes': GaussianNB()
    }

    results = []

    for name, model in models.items():
        print(f"\n正在训练 {name}...")
        try:
            # 训练时间
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train

            # 预测时间
            start_pred = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_pred

            # 评估指标
            acc = accuracy_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # 交叉验证
            cv_acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

            results.append({
                'Model': name,
                'Accuracy': acc,
                'Recall': rec,
                'F1 Score': f1,
                'CV Accuracy': cv_acc,
                'Train Time (s)': train_time,
                'Pred Time (s)': pred_time
            })

            print(f"{name} 训练完成. 准确率: {acc:.4f}")
        except Exception as e:
            print(f"训练 {name} 时出错: {str(e)}")
            continue

    return pd.DataFrame(results)


# 4) 特征筛选
def select_features(df):
    """选择最重要的10个特征"""
    # 这里我们手动选择10个与恶意流量分类高度相关的特征
    selected_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean'
    ]

    # 检查这些特征是否存在于数据中
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = set(selected_features) - set(available_features)

    if missing_features:
        print(f"\n警告: 以下特征不存在于数据中: {missing_features}")

    print(f"\n最终选择的特征({len(available_features)}个): {available_features}")

    # 返回筛选后的数据集
    if 'label' in df.columns:
        return df[available_features + ['label']]
    else:
        return df[available_features]


# 主流程
def main():
    try:
        # 1. 数据读取与预处理
        print("Loading and preprocessing data...")
        df, label_encoder = load_and_preprocess_data()

        # 2. 数据集划分
        X_train, X_test, y_train, y_test = split_data(df)

        # 3. 使用全部特征训练和评估模型
        print("\nTraining models with all features...")
        full_feature_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        print("\nResults with all features:")
        print(full_feature_results.to_markdown())

        # 4. 特征筛选
        print("\nSelecting top 10 features...")
        df_selected = select_features(df)
        X_train_sel, X_test_sel, y_train_sel, y_test_sel = split_data(df_selected)

        # 5. 使用筛选后的特征训练和评估模型
        print("\nTraining models with selected features...")
        selected_feature_results = train_and_evaluate_models(X_train_sel, X_test_sel, y_train_sel, y_test_sel)
        print("\nResults with selected features:")
        print(selected_feature_results.to_markdown())

        # 6. 性能对比分析
        print("\nPerformance comparison:")
        comparison = pd.merge(
            full_feature_results.add_prefix('Full_'),
            selected_feature_results.add_prefix('Sel_'),
            left_on='Full_Model',
            right_on='Sel_Model'
        )
        print(comparison[['Full_Model', 'Full_Accuracy', 'Sel_Accuracy',
                          'Full_F1 Score', 'Sel_F1 Score',
                          'Full_Train Time (s)', 'Sel_Train Time (s)']].to_markdown())

    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()