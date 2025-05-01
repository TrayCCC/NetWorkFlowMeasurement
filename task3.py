import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # 添加这行导入
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score
import time
import warnings

warnings.filterwarnings('ignore')


# 1. 数据预处理（完整实现）
def load_and_preprocess_data():
    # 读取四个CSV文件并添加标签
    data_dir = 'dataSetCsv/'
    try:
        adware = pd.read_csv(f'{data_dir}adware.csv', encoding='latin1')
        adware['label'] = 'adware'

        normal = pd.read_csv(f'{data_dir}normal.csv', encoding='latin1')
        normal['label'] = 'normal'

        trojan = pd.read_csv(f'{data_dir}trojan.csv', encoding='latin1')
        trojan['label'] = 'trojan'

        viaxmr = pd.read_csv(f'{data_dir}viaxmr.csv', encoding='latin1')
        viaxmr['label'] = 'viaxmr'
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise

    # 合并数据集
    df = pd.concat([adware, normal, trojan, viaxmr], axis=0, ignore_index=True)
    print(f"初始数据量: {len(df)}条记录")

    # 删除字符类型特征
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        raise ValueError("没有找到数值型特征列")

    df = df[numeric_cols.append(pd.Index(['label']))]
    print(f"选择数值特征后数据量: {len(df)}条记录")

    # 数据清洗
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"删除{initial_count - len(df)}条完全重复的记录")

    # 处理缺失值
    df = df.dropna(axis=1, how='all')
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"处理缺失值后数据量: {len(df)}条记录")

    # 异常值处理
    for col in numeric_cols:
        if col in df.columns and df[col].nunique() > 1:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"处理异常值后数据量: {len(df)}条记录")

    if len(df) == 0:
        raise ValueError("数据清洗后没有剩余数据")

    # 归一化处理
    scaler = MinMaxScaler()
    features = df.drop('label', axis=1)
    df[features.columns] = scaler.fit_transform(features)

    # 标签编码
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    return df, le


# 2. 数据集划分
def split_data(df):
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\n数据集划分结果:")
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# 3. 模型训练与评估
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # 基础模型
    base_models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }

    # 集成策略
    knn = KNeighborsClassifier()
    svm = SVC(probability=True, random_state=42)
    nb = GaussianNB()

    ensemble_models = {
        'Bagging(KNN)': BaggingClassifier(estimator=knn, n_estimators=10, random_state=42),
        'Boosting(KNN)': AdaBoostClassifier(estimator=knn, n_estimators=50, random_state=42),
        'Voting(KNN+SVM+NB)': VotingClassifier(
            estimators=[('knn', knn), ('svm', svm), ('nb', nb)],
            voting='soft')
    }

    results = []

    # 测试基础模型
    for name, model in base_models.items():
        print(f"正在训练 {name}...")
        try:
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train

            start_pred = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_pred

            acc = accuracy_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({
                'Model': name,
                'Type': 'Base',
                'Accuracy': acc,
                'Recall': rec,
                'F1 Score': f1,
                'Train Time (s)': train_time,
                'Pred Time (s)': pred_time
            })
            print(f"{name} 训练完成. 准确率: {acc:.4f}")
        except Exception as e:
            print(f"训练 {name} 时出错: {str(e)}")
            continue

    # 测试集成模型
    for name, model in ensemble_models.items():
        print(f"正在训练 {name}...")
        try:
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train

            start_pred = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_pred

            acc = accuracy_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({
                'Model': name,
                'Type': 'Ensemble',
                'Accuracy': acc,
                'Recall': rec,
                'F1 Score': f1,
                'Train Time (s)': train_time,
                'Pred Time (s)': pred_time
            })
            print(f"{name} 训练完成. 准确率: {acc:.4f}")
        except Exception as e:
            print(f"训练 {name} 时出错: {str(e)}")
            continue

    return pd.DataFrame(results)


# 主流程
def main():
    try:
        # 1. 数据预处理
        print("Loading and preprocessing data...")
        df, label_encoder = load_and_preprocess_data()

        # 2. 数据集划分
        X_train, X_test, y_train, y_test = split_data(df)

        # 3. 模型训练与评估
        print("\nTraining and evaluating models...")
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

        # 4. 结果分析
        print("\n=== 完整结果 ===")
        print(results.to_markdown())

        # 对比决策树和随机森林
        print("\n=== 决策树 vs 随机森林 ===")
        dt_rf = results[results['Model'].isin(['Decision Tree', 'Random Forest'])]
        print(dt_rf.to_markdown())

        # 对比基础模型和集成模型
        print("\n=== 基础模型 vs 集成模型 ===")
        base_vs_ensemble = results.groupby('Type').mean(numeric_only=True)[
            ['Accuracy', 'Recall', 'F1 Score', 'Train Time (s)', 'Pred Time (s)']]
        print(base_vs_ensemble.to_markdown())

        # 比较不同集成策略
        print("\n=== 不同集成策略比较 ===")
        ensemble_only = results[results['Type'] == 'Ensemble']
        print(ensemble_only.to_markdown())

    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()