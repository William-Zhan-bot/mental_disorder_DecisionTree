# mental_disorder_DecisionTree
以機器學習決策樹算法，針對精神疾病的診斷資料進行研究與檢驗。比較與實際心理衡鑑、精神疾病診斷的異同。

最近在Kaggle上面，找到了有關心理疾病有關的資料集，內容是有關於身心上面的症狀，以及相對應的精神醫學診斷。

在台灣，心理師不能向精神科醫師，開立具有醫學證明的疾病診斷，但是一般在大學、研究所階段，因為心理疾病與衡鑑屬於心理學、心理治療相關的領域，也為了在往後的職涯中與其他系統合作，還是會進行精神疾病、變態心理等相關的訓練，國考的心理師證照，也要求對這門專業要有一定的掌握。

作者身為心理相關科系的畢業生，於碩士階段與大學階段，皆修習了'變態心理學'、'變態心理學研究'、'心理衡鑑研究'等等的課程。對於精神疾病與相關的衡鑑、診斷過程，有基本的了解與認識，因此想以心理學專業的視角，檢視以決策樹(Decision Tree)的方式，會如何根據當事人的症狀，進行精神疾病的診斷，進行區分的主要特徵為何? 電腦所得的結果是否和一般的心理衡鑑、臨床診斷麼模式有差異。

資料集:
https://www.kaggle.com/datasets/cid007/mental-disorder-classification/data


我們先導入資料並進行簡單的資料清洗

```python
# 導入Library
import numpy as np
import pandas as pd
```

```python
# open files
df = pd.read_csv('Dataset-Mental-Disorders.csv')
df.head()

# 印出資料集當中的columns
print(df.columns)
print(len(df.columns))
```

這邊放一個圖片大圖

我們可以看到說，這個資料集包含了從Patient Number到Expert Diagnose等19個屬性。以下為詳細的19個屬性一覽，附上翻譯後面會用到。

"Patient Number": "病人編號"
"Sadness": "悲傷"
"Euphoric": "狂喜"
"Exhausted": "筋疲力盡"
"Sleep dissorder": "睡眠障礙"
"Mood Swing": "情緒波動"
"Suicidal thoughts": "自殺念頭"
"Anorxia": "厭食症"
"Authority Respect": "尊重權威"
"Try-Explanation": "嘗試解釋"
"Aggressive Response": "積極回應"
"Ignore & Move-On": "忽略並繼續前進"
"Nervous Break-down": "神經崩潰"
"Admit Mistakes": "承認錯誤"
"Overthinking": "過度思考"
"Sexual Activity": "性行為"
"Concentration": "注意力"
"Optimisim": "樂觀主義"
"Expert Diagnose": "專家診斷"

可以看到除了第一個的病人編號之外，其他的都是有關個人症狀的描述，而最後一個"Expert Diagnose"則屬於預設的診斷結果。
除去第一個的病人編號與最後一個的專家診斷，共有17個feature可以進行預測。

```python
a=df['Expert Diagnose'].value_counts()
item_counts = pd.Series(a)
# 使用 index 屬性取得項目的名稱
item_names = item_counts.index.tolist()
print(item_names)
```
診斷結果如下，分別為: Normal無症狀、Bipolar Type-1 型1雙極情感症候群、Bipolar Type-2 型2雙極情感症候群、Depression憂鬱症。後續依據轉成0、1、2、3的label。
因為有4個Label，這會屬於多元分類問題。
```
['Bipolar Type-2', 'Depression', 'Normal', 'Bipolar Type-1']
```

資料清洗
```python
# 資料清洗 刪除病人編號
df = df.drop(columns=['Patient Number'])
cols=list(df.columns)

# 檢驗有無遺漏值 (本資料集無)
for col in cols:
    missing_values=df[col].isnull().sum()
    print(f'{col}項目的缺失值為: {missing_values}')

```

檢查特徵項目，已進行重新編碼與替換
```python
for col in cols:
    print(df[col].unique())
```
項目的描述分成3種:
行為有無: YES NO
行為頻率: Sometimes' 'Usually' 'Seldom' 'Most-Often'
行為強度: '1~10 From 10' 
其中YES NO有一行YES多一個空格需特殊處理

進行相對應的轉換與資料處理
```python
wash={
'1 From 10': 1, '2 From 10': 2, '3 From 10': 3, '4 From 10': 4, 
'5 From 10': 5, '6 From 10': 6, '7 From 10': 7, '8 From 10': 8, 
'9 From 10': 9, '10 From 10': 10,
'NO':0, 'YES':1,'YES ':1,
'Usually':3, 'Sometimes':2, 
'Seldom':1, 'Most-Often':4,
'Bipolar Type-2':2, 'Depression':3, 
'Bipolar Type-1':1, 'Normal':0,
}

# 轉換標記資料
for column in df.columns:
    df[column]= df[column].replace(wash)

# 轉換1~10頻率
temp={}
for i in range(1,11):
    temp[str(str(i)+' From 10')]=int(i)

df.head()
```

結果展示02

完成資料的清洗過後，我們先拆分資料以利後續分析
80% train case
20% test case
```python
# 拆分資料
from sklearn.model_selection import train_test_split
x=df.drop(columns='Expert Diagnose')
y=df['Expert Diagnose']
train_x,val_x,train_y,val_y=train_test_split(x, y, test_size=0.2, random_state=87)
```
設定決策樹
```python
# 決策樹
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# 超參數
param_grid = {
    'max_depth': [3, 5, 7, 10],  # 樹的最大深度
    'min_samples_split': [2, 5, 10],  # 節點分裂所需的最小樣本數
    'min_samples_leaf': [1, 2, 4],  # 葉子節點所需的最小樣本數
}

# 初始化
dctree= DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=dctree, param_grid=param_grid, cv=5)

# 在訓練數據上進行網格搜索
grid_search.fit(train_x, train_y)

# 獲取最佳參數
best_params = grid_search.best_params_
print("最佳參數:", best_params)

# 使用最佳參數初始化模型
best_dctree = DecisionTreeClassifier(**best_params)

best_dctree.fit(train_x, train_y)

```
最佳參數: {'max_depth': 3, 'min_samples_leaf': 4, 'min_samples_split': 2}

```python
# testcase預測
X=best_dctree.predict(train_x)
accuracy=accuracy_score(X, train_y)
print('原本預測分數:',accuracy)

# 進行樹的預測
tree_pred_y=best_dctree.predict(val_x)

# 結果檢驗
accuracy=accuracy_score(tree_pred_y, val_y)
print('決策樹預測分數:', accuracy)
```

```python
from sklearn.tree import export_graphviz
import graphviz
# 將決策樹導出為Graphviz格式
feature_names=['悲傷','歡欣','疲憊',
'睡眠障礙','心情波動',
'自殺念頭','厭食症',
'尊重權威','嘗試解釋',
'攻擊性回應','忽略並繼續前進','神經崩潰',
'承認錯誤','過度思考','性活動',
'專注力','樂觀']
class_names=['正常','躁鬱-1','躁鬱-2','憂鬱']
dot_data = export_graphviz(best_dctree, out_file=None, 
                           feature_names=feature_names,  
                           class_names=class_names,  
                           filled=True, rounded=True,  
                           special_characters=True)


# 使用Graphviz將Graphviz格式轉換為圖片
graph = graphviz.Source(dot_data) 

# 保存圖片到文件
graph.render("D:/desktop/disorder_decision_tree")

graph

```
決策樹圖片



