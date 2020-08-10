# Jeju Credit Card Usage Prediction - July

* The REG_YYMM may not be as important: the date cannot reflect anything in this model.


```python
import pandas as pd
import numpy as np 
import os
import itertools
from tqdm.notebook import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from pandas.plotting import scatter_matrix
from sklearn import preprocessing, metrics
import category_encoders as ce

"""
For model building, consider
- XGboost Regressor
- Random Forest Regressor
- Linear Regressor
- SVM

"""
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
import warnings

warnings.filterwarnings("ignore")

# 한글 글씨체 깨짐 방지
plt.rcParams['font.family'] = 'NanumBarunGothic'

%matplotlib inline
```


```python

# Read files 
train_data = pd.read_csv('201901-202003.csv')
apr_data = pd.read_csv('202004.csv')
submission = pd.read_csv('submission.csv')

# Append april data to test data
train_data = train_data.append(apr_data)

# fill all empty entries with ''
train_data = train_data.fillna('')

# Drop unnecessary features - 소비자의 데이터는 그다지 중요하지 않음
train_data.drop(['HOM_SIDO_NM','HOM_CCG_NM'], axis=1, inplace=True)

# Rename SEX 
train_data.rename(columns={'SEX_CTGO_CD':'SEX'}, inplace=True)

# change data type of SEX
train_data['SEX'] = train_data['SEX'].astype(str)
```


```python
#train_data.info()
#train_data.isna().sum()
train_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REG_YYMM</th>
      <th>CARD_SIDO_NM</th>
      <th>CARD_CCG_NM</th>
      <th>STD_CLSS_NM</th>
      <th>AGE</th>
      <th>SEX</th>
      <th>FLC</th>
      <th>CSTMR_CNT</th>
      <th>AMT</th>
      <th>CNT</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# Make dict of {SIDO: [CCG]} - 각 도에 포함된 시 
SIDO_CCG_dict = {}

# List of all unique CARD_SIDO_NM
sidos = list(train_data['CARD_SIDO_NM'].unique())

for sido in sidos:
    ccgs = list(train_data[train_data['CARD_SIDO_NM'] == sido]['CARD_CCG_NM'].unique())
    SIDO_CCG_dict[sido] = ccgs

```


```python
# Group by and keep only useful features
train_2 = pd.DataFrame(train_data.groupby(['CARD_CCG_NM','STD_CLSS_NM','AGE','SEX'])['AMT'].sum())
train_2.reset_index(inplace=True)
train_x = train_2[['CARD_CCG_NM','STD_CLSS_NM','AGE','SEX']]
train_y = pd.DataFrame(np.log(train_2['AMT']))

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AMT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.789685</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16.925373</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17.022719</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17.638633</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.459034</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>82851</th>
      <td>18.238076</td>
    </tr>
    <tr>
      <th>82852</th>
      <td>17.498859</td>
    </tr>
    <tr>
      <th>82853</th>
      <td>16.085273</td>
    </tr>
    <tr>
      <th>82854</th>
      <td>16.190505</td>
    </tr>
    <tr>
      <th>82855</th>
      <td>14.694182</td>
    </tr>
  </tbody>
</table>
<p>82856 rows × 1 columns</p>
</div>




```python
# Now keep the test_data in the same format
# Keep only REG_YYMM = 202007
test_data = submission[submission['REG_YYMM'] == 202007]

test_x = test_data[['CARD_SIDO_NM','STD_CLSS_NM']]
test_y = test_data['AMT']

# Make the test_x include CARD_CCG_NM as well
columns = ['CARD_SIDO_NM','CARD_CCG_NM']
SIDO_CCG = pd.DataFrame(columns=columns)

for sido in sidos:
    for ccg in SIDO_CCG_dict[sido]:
        SIDO_CCG = SIDO_CCG.append({'CARD_SIDO_NM':sido, 'CARD_CCG_NM':ccg}, ignore_index=True)
        
test_x = pd.merge(test_x, SIDO_CCG)

test_sido_ccg = test_x[['CARD_SIDO_NM','CARD_CCG_NM']]

# Add SEX column
temp_sex = pd.DataFrame({'SEX':['1','2']})
temp_sex['key'] = 0
test_x['key'] = 0

test_x = test_x.merge(temp_sex, on='key')

# Add AGE column
temp_age = pd.DataFrame({'AGE':['10s','20s','30s','40s','50s','60s','70s']})
temp_age['key'] = 0
test_x = test_x.merge(temp_age, on='key')

test_x.drop(['CARD_SIDO_NM','key'], axis=1, inplace=True)
test_x
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>STD_CLSS_NM</th>
      <th>CARD_CCG_NM</th>
      <th>SEX</th>
      <th>AGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>건강보조식품 소매업</td>
      <td>강릉시</td>
      <td>1</td>
      <td>10s</td>
    </tr>
    <tr>
      <th>1</th>
      <td>건강보조식품 소매업</td>
      <td>강릉시</td>
      <td>1</td>
      <td>20s</td>
    </tr>
    <tr>
      <th>2</th>
      <td>건강보조식품 소매업</td>
      <td>강릉시</td>
      <td>1</td>
      <td>30s</td>
    </tr>
    <tr>
      <th>3</th>
      <td>건강보조식품 소매업</td>
      <td>강릉시</td>
      <td>1</td>
      <td>40s</td>
    </tr>
    <tr>
      <th>4</th>
      <td>건강보조식품 소매업</td>
      <td>강릉시</td>
      <td>1</td>
      <td>50s</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>143495</th>
      <td>휴양콘도 운영업</td>
      <td>충주시</td>
      <td>2</td>
      <td>30s</td>
    </tr>
    <tr>
      <th>143496</th>
      <td>휴양콘도 운영업</td>
      <td>충주시</td>
      <td>2</td>
      <td>40s</td>
    </tr>
    <tr>
      <th>143497</th>
      <td>휴양콘도 운영업</td>
      <td>충주시</td>
      <td>2</td>
      <td>50s</td>
    </tr>
    <tr>
      <th>143498</th>
      <td>휴양콘도 운영업</td>
      <td>충주시</td>
      <td>2</td>
      <td>60s</td>
    </tr>
    <tr>
      <th>143499</th>
      <td>휴양콘도 운영업</td>
      <td>충주시</td>
      <td>2</td>
      <td>70s</td>
    </tr>
  </tbody>
</table>
<p>143500 rows × 4 columns</p>
</div>



# Feature Generation
* There are now only 4 features in our train_x. (STD_CLSS_NM, CARD_CCG_NM, SEX, AGE)
* Combine each feature, and use target encodings to convert categorical variables into numerical variables
* Both on train and test 


```python

def feature_combine(df_1, df_2):
    df_train = df_1.copy()
    df_test = df_2.copy()
    cat_features = df_train.columns.tolist()
    # Iterate through cat_features into 10 different combinations
    for features in itertools.combinations(cat_features, 2):
        new_feature = features[0] + '_' + features[1]
        # Make combined column
        df_train[new_feature] = df_train[features[0]] + '_' + df_train[features[1]]
        df_test[new_feature] = df_test[features[0]] + '_' + df_test[features[1]]
        
        # Groupby 
        df_grouped = pd.DataFrame(train_data.groupby([features[0],features[1]])['AMT'].mean())
        df_grouped.reset_index(inplace=True)
        df_grouped[new_feature] = df_grouped[features[0]] + '_' + df_grouped[features[1]]
        encoder = pd.Series(np.log(df_grouped['AMT'].values), index=df_grouped[new_feature])
        
        # Encoding process
        df_train[new_feature] = df_train[new_feature].map(encoder)
        df_test[new_feature] = df_test[new_feature].map(encoder)
        
    return df_train, df_test


train_final_x, test_final_x = feature_combine(train_x, test_x)
```


```python

# Label Encodings for AGE 
train_final_x['SEX'] = train_final_x['SEX'].astype('int64')
test_final_x['SEX'] = test_final_x['SEX'].astype('int64')

# AGE ordinal encoding
age_dict = {'10s':1,'70s':2, '20s':3, '60s':4, '30s':5, '40s':6, '50s':7}
train_final_x['AGE'] = train_final_x['AGE'].apply(lambda x: age_dict[x])
test_final_x['AGE'] = test_final_x['AGE'].apply(lambda x: age_dict[x])

# STD_CLSS and CARD_CCG target encoding
for feature in ['STD_CLSS_NM','CARD_CCG_NM']:
    temp_group = np.log(train_data.groupby([feature])['AMT'].mean())
    train_final_x[feature + '_encoded'] = train_final_x[feature].map(temp_group)
    test_final_x[feature + '_encoded'] = test_final_x[feature].map(temp_group)
    
train_final_x.drop(['CARD_CCG_NM','STD_CLSS_NM'], axis=1, inplace=True)
test_final_x.drop(['CARD_CCG_NM','STD_CLSS_NM'], axis=1, inplace=True)

```


```python
# Check any null value in the test_x entry 
test_final_x.isna().sum()
test_final_x.fillna(0, inplace=True)
test_final_x.isna().sum()
```




    SEX                        0
    AGE                        0
    CARD_CCG_NM_STD_CLSS_NM    0
    CARD_CCG_NM_AGE            0
    CARD_CCG_NM_SEX            0
    STD_CLSS_NM_AGE            0
    STD_CLSS_NM_SEX            0
    AGE_SEX                    0
    STD_CLSS_NM_encoded        0
    CARD_CCG_NM_encoded        0
    dtype: int64




```python
# Check column match
print(train_final_x.shape, test_final_x.shape)

assert(train_final_x.columns.all() == test_final_x.columns.all())
```

    (82856, 10) (143500, 10)


# EDA 
* Check correlation among features


```python
# Now check correlation between each input variable and target
corr_df = pd.concat([train_final_x,train_y], axis=1)
corr = corr_df.corr()
display(corr.style.background_gradient(cmap='coolwarm').set_precision(4))
print(corr['AMT'].sort_values(ascending=False))

```




    AMT                        1.000000
    STD_CLSS_NM_AGE            0.595775
    CARD_CCG_NM_STD_CLSS_NM    0.581253
    STD_CLSS_NM_SEX            0.435275
    STD_CLSS_NM_encoded        0.410177
    CARD_CCG_NM_AGE            0.401049
    AGE_SEX                    0.334909
    AGE                        0.334214
    CARD_CCG_NM_SEX            0.282897
    CARD_CCG_NM_encoded        0.273854
    SEX                       -0.054095
    Name: AMT, dtype: float64



```python
# Model selection and prediction 

model_1 = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, verbose=10)
model_1.fit(train_final_x, train_y)
print("Fit Complete")
prediction = model_1.predict(test_final_x)

"""
#model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=-1, verbose=10)
#model_2.fit(train_x, train_y)
#print("Fit Complete")
prediction = model_2.predict(apr_x)
"""
```

    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 4 concurrent workers.


    building tree 1 of 50building tree 2 of 50building tree 3 of 50
    building tree 4 of 50
    
    
    building tree 5 of 50
    building tree 6 of 50
    building tree 7 of 50
    building tree 8 of 50


    [Parallel(n_jobs=-1)]: Done   5 tasks      | elapsed:    1.8s


    building tree 9 of 50
    building tree 10 of 50
    building tree 11 of 50
    building tree 12 of 50
    building tree 13 of 50
    building tree 14 of 50
    building tree 15 of 50
    building tree 16 of 50


    [Parallel(n_jobs=-1)]: Done  10 tasks      | elapsed:    2.9s


    building tree 17 of 50
    building tree 18 of 50
    building tree 19 of 50
    building tree 20 of 50
    building tree 21 of 50
    building tree 22 of 50
    building tree 23 of 50


    [Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed:    4.5s


    building tree 24 of 50
    building tree 25 of 50
    building tree 26 of 50
    building tree 27 of 50
    building tree 28 of 50


    [Parallel(n_jobs=-1)]: Done  24 tasks      | elapsed:    5.9s


    building tree 29 of 50
    building tree 30 of 50
    building tree 31 of 50
    building tree 32 of 50
    building tree 33 of 50
    building tree 34 of 50
    building tree 35 of 50
    building tree 36 of 50


    [Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed:    8.1s


    building tree 37 of 50
    building tree 38 of 50
    building tree 39 of 50
    building tree 40 of 50
    building tree 41 of 50
    building tree 42 of 50
    building tree 43 of 50
    building tree 44 of 50
    building tree 45 of 50
    building tree 46 of 50
    building tree 47 of 50

    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   10.5s


    
    building tree 48 of 50
    building tree 49 of 50
    building tree 50 of 50


    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   12.3s finished
    [Parallel(n_jobs=4)]: Using backend ThreadingBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done   5 tasks      | elapsed:    0.0s


    Fit Complete


    [Parallel(n_jobs=4)]: Done  10 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=4)]: Done  17 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=4)]: Done  24 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=4)]: Done  33 tasks      | elapsed:    0.3s
    [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:    0.4s
    [Parallel(n_jobs=4)]: Done  50 out of  50 | elapsed:    0.5s finished





    '\n#model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=-1, verbose=10)\n#model_2.fit(train_x, train_y)\n#print("Fit Complete")\nprediction = model_2.predict(apr_x)\n'




```python
prediction = pd.DataFrame(np.exp(prediction))
prediction.columns = ['AMT']
final_df = pd.concat([test_x, prediction], axis=1)

# Add CARD_SIDO column
final_df = final_df.merge(test_sido_ccg, how='inner', on='CARD_CCG_NM')
final_df.drop_duplicates(inplace=True)
final_df = pd.DataFrame(final_df.groupby(['CARD_SIDO_NM','STD_CLSS_NM'])['AMT'].sum())
final_df.reset_index(inplace=True)
final_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CARD_SIDO_NM</th>
      <th>STD_CLSS_NM</th>
      <th>AMT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>강원</td>
      <td>건강보조식품 소매업</td>
      <td>2.019416e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>강원</td>
      <td>골프장 운영업</td>
      <td>7.598521e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>강원</td>
      <td>과실 및 채소 소매업</td>
      <td>5.291839e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>강원</td>
      <td>관광 민예품 및 선물용품 소매업</td>
      <td>1.624260e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>강원</td>
      <td>그외 기타 분류안된 오락관련 서비스업</td>
      <td>2.650708e+07</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>692</th>
      <td>충북</td>
      <td>피자 햄버거 샌드위치 및 유사 음식점업</td>
      <td>3.585832e+09</td>
    </tr>
    <tr>
      <th>693</th>
      <td>충북</td>
      <td>한식 음식점업</td>
      <td>3.328297e+10</td>
    </tr>
    <tr>
      <th>694</th>
      <td>충북</td>
      <td>호텔업</td>
      <td>1.858286e+08</td>
    </tr>
    <tr>
      <th>695</th>
      <td>충북</td>
      <td>화장품 및 방향제 소매업</td>
      <td>3.955902e+09</td>
    </tr>
    <tr>
      <th>696</th>
      <td>충북</td>
      <td>휴양콘도 운영업</td>
      <td>1.402585e+08</td>
    </tr>
  </tbody>
</table>
<p>697 rows × 3 columns</p>
</div>




```python
# Now fit the data into submission 
# First, groupby for REG_YYMM == 202004
apr_data = pd.DataFrame(apr_data.groupby(['REG_YYMM','CARD_SIDO_NM','STD_CLSS_NM'])['AMT'].sum())
apr_data.reset_index(inplace=True)
display(apr_data)

# Reinitialize 'AMT' column to 0 before insertion
submission['AMT'] = 0

# Fill in 
submission_final = submission.merge(apr_data, how='left', on=['REG_YYMM','STD_CLSS_NM','CARD_SIDO_NM'])
submission_final.drop(['id','AMT_x'], axis=1, inplace=True)
submission_final.rename(columns={'AMT_y':'AMT'}, inplace=True)
submission_final.loc[submission_final['REG_YYMM'] == 202007, 'AMT'] = final_df['AMT'].values
submission_final.fillna(0, inplace=True)

submission_final.to_csv('submission.csv', encoding='UTF-8-sig')
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REG_YYMM</th>
      <th>CARD_SIDO_NM</th>
      <th>STD_CLSS_NM</th>
      <th>AMT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>202004</td>
      <td>강원</td>
      <td>건강보조식품 소매업</td>
      <td>88823988</td>
    </tr>
    <tr>
      <th>1</th>
      <td>202004</td>
      <td>강원</td>
      <td>골프장 운영업</td>
      <td>4708346820</td>
    </tr>
    <tr>
      <th>2</th>
      <td>202004</td>
      <td>강원</td>
      <td>과실 및 채소 소매업</td>
      <td>1121028924</td>
    </tr>
    <tr>
      <th>3</th>
      <td>202004</td>
      <td>강원</td>
      <td>관광 민예품 및 선물용품 소매업</td>
      <td>14360780</td>
    </tr>
    <tr>
      <th>4</th>
      <td>202004</td>
      <td>강원</td>
      <td>그외 기타 스포츠시설 운영업</td>
      <td>227200</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>605</th>
      <td>202004</td>
      <td>충북</td>
      <td>피자 햄버거 샌드위치 및 유사 음식점업</td>
      <td>1373635928</td>
    </tr>
    <tr>
      <th>606</th>
      <td>202004</td>
      <td>충북</td>
      <td>한식 음식점업</td>
      <td>18911036160</td>
    </tr>
    <tr>
      <th>607</th>
      <td>202004</td>
      <td>충북</td>
      <td>호텔업</td>
      <td>14121500</td>
    </tr>
    <tr>
      <th>608</th>
      <td>202004</td>
      <td>충북</td>
      <td>화장품 및 방향제 소매업</td>
      <td>450507431</td>
    </tr>
    <tr>
      <th>609</th>
      <td>202004</td>
      <td>충북</td>
      <td>휴양콘도 운영업</td>
      <td>9328420</td>
    </tr>
  </tbody>
</table>
<p>610 rows × 4 columns</p>
</div>

