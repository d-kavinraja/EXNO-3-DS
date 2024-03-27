## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```py
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

![output](./output/o1.png)

```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![output](./output/o2.png)

```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![output](./output/o3.png)

```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![output](./output/o4.png)

```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

![Screenshot 2024-03-26 161716](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/4d93974b-16f9-4b27-9d00-4cc0abee08d0)

```py
df2=pd.concat([df2,enc],axis=1)
df2
```

![Screenshot 2024-03-26 161722](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/74c5c492-dd39-4f11-a75a-64f6a28707bc)

```py
pd.get_dummies(df2,columns=["nom_0"])
```

![Screenshot 2024-03-26 161729](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/b39577a2-d937-4e22-8623-e19735380801)

```py
pip install --upgrade category_encoders
```
![Screenshot 2024-03-26 162300](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/3066f410-80e3-4069-be2d-4b056d132b48)

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

![Screenshot 2024-03-26 162308](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/fdbc718b-825b-47ea-8629-31c5303a56d2)

```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

![Screenshot 2024-03-26 162313](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/973be76c-caf8-4036-8428-ebeb2c035178)

```py
dfb=pd.concat([df,nd],axis=1)
dfb
```

![Screenshot 2024-03-26 162319](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/9b0226ab-1692-4377-bce1-e781dd2b05de)

```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![Screenshot 2024-03-26 162325](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/b89fd4dc-c843-489b-8fb1-c4cbece4b686)

```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![Screenshot 2024-03-26 162333](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/993b0cab-3503-4b86-831e-f661df5b5ca3)

```py
df.skew()
```

![Screenshot 2024-03-26 164950](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/2ee3d99f-b61f-4ec0-ba75-65411a38c4d7)

```py
np.log(df["Highly Positive Skew"])
```

![Screenshot 2024-03-26 165006](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/c21948d7-8891-4a5c-b3e2-d60add4cac7a)

```py
np.reciprocal(df["Moderate Positive Skew"])
```

![Screenshot 2024-03-26 165013](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/2f1432cd-a9ee-4f32-9da1-3a2808d47752)

```py
np.sqrt(df["Highly Positive Skew"])
```

![Screenshot 2024-03-26 165017](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/b9927ce7-1ffd-4320-98de-0ce9a340e20e)

```py
np.square(df["Highly Positive Skew"])
```

![Screenshot 2024-03-26 165022](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/90e9195f-7fb3-46fb-b085-cd5fdbe95c82)

```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![Screenshot 2024-03-26 165027](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/b122883e-8a6e-4779-bc85-09886fb6de90)

```py
df.skew()
```

![Screenshot 2024-03-26 165032](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/bfbec92b-dc15-4238-8eae-f2ce3be43bc0)

```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```

![Screenshot 2024-03-26 165037](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/a078d4cb-dce2-45a6-b613-99cc4d313966)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![Screenshot 2024-03-26 165052](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/f33b4561-b92d-4f18-967e-e3224d7a4a28)

```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![Screenshot 2024-03-26 165056](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/fae1ae22-a0b7-4fbb-b40e-b8436fe9a11f)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![Screenshot 2024-03-26 165106](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/bb53f1a5-7d35-414d-b196-24eca3434780)

```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![Screenshot 2024-03-26 165111](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/cc2d9ce7-4811-44a0-bf21-aeed274b2728)

```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![Screenshot 2024-03-26 165117](https://github.com/gokulapriya632202/EXNO-3-DS/assets/119560302/c6b4c14b-1d8d-4802-9651-1cc7cb3a610f)


# RESULT:
 Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
