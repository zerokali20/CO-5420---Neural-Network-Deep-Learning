import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
#Loading Dataset
diabetes_df = pd.read_csv('diabetes.csv')
diabetes_df.head() # Preview the dataset
diabetes_df.shape # Number of instances and variables
#Renaming columns
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
diabetes_df.columns = col_names # Rename column names
diabetes_df.info()    #Summary of dataset
#Frequency distributions of values in variables
for col in col_names:
    print(diabetes_df[col].value_counts())
diabetes_df['label'].value_counts()  #Exploring target variable
diabetes_df.isnull().sum() #Checking missing values in variables
X = diabetes_df.drop(['label'], axis=1)
y = diabetes_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1) # 75% training and 25% test
X_train.shape, X_test.shape # Shapes of X_train and X_test
X_train.dtypes # Check data types in X_train
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=X.columns.tolist())
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=0)
clf_gini.fit(X_train, y_train) # Train the classifier
y_pred = clf_gini.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(
    clf_gini,
    out_file=dot_data,
    filled=True,
    rounded=True,
    special_characters=True,
    feature_names=X.columns,
    class_names=['0', '1']
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())