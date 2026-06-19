# E-COMMERCE DATASET 

import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.metrics import (accuracy_score,f1_score,roc_auc_score,roc_curve,
                             precision_score,recall_score,confusion_matrix,classification_report)
df = pd.read_csv(r"C:\Users\Dell\Downloads\archive (3).zip")
print (df.duplicated().sum())
print (df.isnull().sum())
print (df.head(5))
print (df.shape)

# DATA ENGINEERING 
for col in df.select_dtypes(include= ['object']).columns.tolist():
    df[col] = df[col].str.lower().str.strip()

print (df['Reached.on.Time_Y.N'].value_counts())
print (df['Reached.on.Time_Y.N'].value_counts(normalize= True))
print ("\nshape:",df.shape)
df=df.drop('ID',axis= 1)
#df = df.rename(columns ={'Reached.on.Time_Y.N':'reaching_time'})


# Feature Engineering
# 1. Kya discount bahut zyada hai?
# Zyada discount = cheap product = careless handling?
df['high_discount'] = (df['Discount_offered'] > 40).astype(int)

# 2. Kya bohot calls aaye? — frustrated customer
df['high_calls'] = (df['Customer_care_calls'] > 4).astype(int)

# 3. Ship mode + Product importance combo
# Road pe low importance = late hone ka chance zyada
df['road_low'] = (
    (df['Mode_of_Shipment'] == 'road') &
    (df['Product_importance'] == 'low')
).astype(int)

# 4. Heavy aur cheap — careless handle hoga
df['heavy_cheap'] = (
    (df['Weight_in_gms'] > 4000) &
    (df['Cost_of_the_Product'] < 150)
).astype(int)

# 5. Prior purchases kam — naya customer — late delivery ka risk
df['new_customer'] = (df['Prior_purchases'] <= 2).astype(int)
# 1. Cost per gram
df['cost_per_gram'] = df['Cost_of_the_Product'] / df['Weight_in_gms']

# 2. Call to rating ratio
df['call_rating_ratio'] = df['Customer_care_calls'] / df['Customer_rating']

# 3. Discount category (Low / Medium / High)
df['discount_category'] = pd.cut(
    df['Discount_offered'],
    bins=[0, 10, 20, df['Discount_offered'].max()],
    labels=['Low', 'Medium', 'High']
)

# 4. Weight category (Light / Medium / Heavy)
df['weight_category'] = pd.cut(
    df['Weight_in_gms'],
    bins=[0, 500, 2000, df['Weight_in_gms'].max()],
    labels=['Light', 'Medium', 'Heavy']
)
 
# 5. High value product (binary flag)
median_cost = df['Cost_of_the_Product'].median()
df['high_value'] = (df['Cost_of_the_Product'] > median_cost).astype(int)


# 7. Discount effectiveness (Discount / Cost)
df['discount_effectiveness'] = df['Discount_offered'] / df['Cost_of_the_Product']

# 8. Weight efficiency (Weight / Cost)
df['weight_efficiency'] = df['Weight_in_gms'] / df['Cost_of_the_Product']

# 9. Customer interaction intensity (Calls * Rating)
df['interaction_intensity'] = df['Customer_care_calls'] * df['Customer_rating']

# 10. Normalized cost (z-score scaling)
df['cost_zscore'] = (df['Cost_of_the_Product'] - df['Cost_of_the_Product'].mean()) / df['Cost_of_the_Product'].std()


df = df.drop(columns=[
    'cost_zscore',        # Cost ka copy
    'weight_efficiency',  # Weight/Cost — already dono hain
    'interaction_intensity' # calls*rating — dono already hain
])
# COLUMNS 

target = 'Reached.on.Time_Y.N'
X = df.drop(columns = [target])
y = df[target]
numeric_cols = df.drop(columns = [target]).select_dtypes(include= ['int64','float64']).columns
categorical_cols = df.select_dtypes(include = ['object']).columns

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size = 0.3)

preprocessor = ColumnTransformer(transformers=(
    ('num',Pipeline([
        ('imputer',SimpleImputer(strategy = 'median')),
        ('scaler',StandardScaler()),
        ]),numeric_cols),

        ('cat',Pipeline([
            ('imputer',SimpleImputer(strategy = 'most_frequent')),
            ('encoder',OneHotEncoder(handle_unknown =  'ignore'))
        ]),categorical_cols)
))

models = {
    'logistic': LogisticRegression(
        class_weight = 'balanced',
        max_iter = 1000,
        random_state = 42
    ),
    'forest':RandomForestClassifier(
        class_weight ='balanced',
        max_depth = 8,
        max_samples = 0.5,
        n_estimators=50,
        random_state =42,
        n_jobs =-1
    ),
    'XGBoost':XGBClassifier(
        eval_metric = 'logloss',
        max_depth = 4,
        n_estimators = 100,
        random_state = 42,
        colsample_bytree=0.8,
        subsample =0.5,
        n_jobs =-1
    )
}
print ("\n"+"="*50)
print (f"{'MODEL':<15} {'ROC-AUC':>12} {'MODEL STD':>12}")
print ("="*50)
results = {}
for name,model in models.items():
    pipe =  Pipeline([
        ('preprocessor',preprocessor),
        ('model',model)
    ])
    scores = cross_val_score(
        pipe,X_train,y_train,cv=5,scoring= 'roc_auc')
    results[name] = scores.mean()
    print (f"{name:<15} {scores.mean():>12.4f} {scores.std():>10.4f}")
print ("="*55)

best = max(results, key =results.get)
print (f"\n best model : {best} ({results[best]:.4f})")
best_pipe= Pipeline([
    ('preprocessor',preprocessor),
    ('model',models[best])
])
best_pipe.fit(X_train,y_train)
pred = best_pipe.predict(X_test)
probs = best_pipe.predict_proba(X_test)[:,1] 

train_score = best_pipe.score(X_train,y_train)
cv_score  = cross_val_score(best_pipe,X_train,y_train,cv=3).mean() 
print (" CHECKING BIAS VS VARINACE :")
print ("train_score>>",train_score)
print ("cv score",cv_score)
print (f" accuracy score is :{accuracy_score(y_test,pred)}")
print (f"classification report is :\n {classification_report(y_test,pred)}")
print (f"confusion matrix is : \n{confusion_matrix(y_test,pred)}")


print ("threshold | accuracy | precision | recall |f1 \n")
for t in np.arange(0.3,0.7,0.1):
    y_t = (probs>=t).astype(int)
    print (f"{t:.2f}  | {accuracy_score(y_test,y_t):.3f}  | "
           f"{precision_score(y_test,y_t):.3f}    |"
           f"{recall_score(y_test,y_t):.3f}   |"
           f"{f1_score(y_test,y_t):.3f}")  
print ("="*65) 

# Best threshold choose karo
print ("="*45)
print ("working on best threshold")
best_threshold = 0.30

# Is threshold pe predict karo
y_final = (probs >= best_threshold).astype(int)

print(f"\n=== FINAL RESULTS AT THRESHOLD {best_threshold} ===")
print(confusion_matrix(y_test, y_final))
print(classification_report(y_test, y_final))
print("F1      :", f1_score(y_test, y_final))
print("ROC-AUC :", roc_auc_score(y_test, probs))
print ("="*45)

from sklearn.ensemble import StackingClassifier

print ("==STACKIGN HO RAHI HAI TIME TAKEN THING===")
base_models = [
    ('logistic',LogisticRegression(
        class_weight = 'balanced',
        max_iter = 1000,
        random_state =42
    )),
    ('forest',RandomForestClassifier(
        class_weight = 'balanced',
        max_depth = 8,
        max_samples =0.5,
        n_estimators=50,
        random_state =42,
        n_jobs= -1
    )),
    ('xgboost',XGBClassifier(
        eval_metric = 'logloss',
        max_depth =4,
        colsample_bytree=0.8,
        n_estimators =100,
        subsample =0.5,
        random_state =42,
        n_jobs =-1
    ))]
meta_model = LogisticRegression(
    class_weight ='balanced',
    max_iter =1000,
    random_state = 42
)
stacking = StackingClassifier(
    estimators = base_models,
    final_estimator=meta_model,
    cv=3,
    stack_method ='predict_proba',
    n_jobs=-1   
)
stacking_pipe = Pipeline([
    ('preprocessor',preprocessor),
    ('model',stacking)
])

stacking_pipe.fit(X_train,y_train)
stack_probs = stacking_pipe.predict_proba(X_test)[:,1]

best_thresh_st = 0.5
best_f1_st =0 
best_recall_st =0

for t in np.arange(0.1,0.91,0.01):
    y_t = (stack_probs >= t).astype(int)
    f1 = f1_score(y_test,y_t)
    rc = recall_score(y_test,y_t)
    if f1 > best_f1_st:
        best_f1_st = f1best_thresh_st =t
        best_recall_st =rc

print(f"\nStacking Best Threshold : {best_thresh_st:.2f}")
print(f"Stacking Best F1        : {best_f1_st:.4f}")
print(f"Stacking Best Recall    : {best_recall_st:.4f}")
y_stack_final = (stack_probs >= best_thresh_st).astype(int)

print(f"\n=== STACKING FINAL AT {best_thresh_st:.2f} ===")
print(confusion_matrix(y_test, y_stack_final))
print(classification_report(y_test, y_stack_final))
print(f"ROC-AUC : {roc_auc_score(y_test, stack_probs):.4f}")

# =======================
# COMPARISON
# =======================
print("\n" + "="*55)
print("FINAL COMPARISON")
print("="*55)
print(f"{'Model':<20} {'ROC-AUC':>10} {'F1':>10} {'Recall':>10}")
print("-"*55)
print(f"{'Forest alone':<20} "
      f"{roc_auc_score(y_test, probs):>10.4f} "
      f"{best_f1_st:>10.4f} "
      f"{recall_score(y_test, y_final):>10.4f}")
print(f"{'Stacking':<20} "
      f"{roc_auc_score(y_test, stack_probs):>10.4f} "
      f"{best_f1_st:>10.4f} "
      f"{best_recall_st:>10.4f}")
print("="*55)

import shap
import numpy as np

print("\nSHAP chal raha hai...")

# Step 1 — transform
X_test_transformed = best_pipe.named_steps['preprocessor'].transform(X_test)
feature_names = list(best_pipe.named_steps['preprocessor'].get_feature_names_out())

# Step 2 — model
forest_model = best_pipe.named_steps['model']

# Step 3 — explainer
explainer = shap.TreeExplainer(forest_model)

# Step 4 — sample lo — fast bhi, clean bhi
X_sample = X_test_transformed[:200]

# Step 5 — shap values
shap_values = explainer.shap_values(X_sample)

# Step 6 — 3D fix — class 1 nikalo
if isinstance(shap_values, list):
    sv = shap_values[1]          # list case
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]    # 3D array case
else:
    sv = shap_values

print("SV shape:", sv.shape)     # ye hona chahiye (200, n_features)

# Step 7 — DataFrame banao
import pandas as pd
X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

# Step 8 — Beeswarm
print("\nBeeswarm plot...")
shap.summary_plot(
    sv,
    X_sample_df,
    max_display=15
)

# Step 9 — Bar plot
print("\nBar plot...")
shap.summary_plot(
    sv,
    X_sample_df,
    plot_type='bar',
    max_display=15
)

# Step 10 — Waterfall
print("\nWaterfall plot...")
shap_exp = shap.Explanation(
    values=sv[0],
    base_values=explainer.expected_value[1],
    data=X_sample[0],
    feature_names=feature_names
)
shap.plots.waterfall(shap_exp)

X_test_df = pd.DataFrame({
    'message':X_test.index,
    "actual":y_test.values,
    'predicted':y_final,
    'probs':probs
})
# wrong predictions
wrong  = X_test_df[X_test_df['actual'] != X_test_df['predicted']]
# wrong with confidence 
wrong_confident = wrong [(wrong['probs']>0.8) | (wrong['probs']<0.2)]
print (f"\n ==HIGH CONFIDENCE WRONG PREDICTIONS==")
print (f"total; {len(wrong_confident)}")
for i, row in wrong_confident.iterrows():
    print (f"\nACTUAL : {'LATE' if row ['actual'] ==1 else 'On-time'}")
    print (f"preidcted : {'Late' if row['predicted']==1 else 'On-time'}")
    print (f"probability : {row['probs']:.3f}")
    print (F"ORDER/MSG : {row['message']}")

fp = X_test_df[(X_test_df['actual']==0)&
               (X_test_df['predicted']==1)]
print  ("===false positive - ham ko spam bola===")
print (f"total fp : {len(fp)}")
for i, row in fp.iterrows():
    print (f"\n prob: {row['probs']:.3f}")
    print (f"\n MSG : {row['message'][:100]}")
print ("\n"+"="*50)

fn = X_test_df[(X_test_df['actual']==1)&
               (X_test_df['predicted']==0)]
print (f"total : {len(fn)}")
for i, row in fn.iterrows():
    print (f"probs : {row['probs']:.3f}")
    print (f" MSG : {row['messgae'][:100]}")
print ("\n"+"="*50)
