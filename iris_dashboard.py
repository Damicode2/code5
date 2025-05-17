{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "669e87c7-d622-4324-90de-b0f94b35168e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-17 12:39:41.769 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\HP\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-05-17 12:39:41.871 Session state does not function when running a script without `streamlit run`\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# iris_dashboard.py\n",
    "\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn import datasets\n",
    "from sklearn.model_selection import train_test_split, KFold, cross_val_score\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.svm import SVC\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "st.title(\"🌼 Iris Dataset Machine Learning Dashboard\")\n",
    "\n",
    "# Load data\n",
    "iris = datasets.load_iris()\n",
    "df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])\n",
    "df['target'] = iris['target']\n",
    "df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})\n",
    "\n",
    "st.subheader(\"1. Dataset Overview\")\n",
    "st.write(df.head())\n",
    "\n",
    "# Sidebar options\n",
    "model_name = st.sidebar.selectbox(\n",
    "    \"Select a Model\",\n",
    "    [\"Logistic Regression\", \"Linear Discriminant Analysis\", \"K-Nearest Neighbors\",\n",
    "     \"Decision Tree\", \"Naive Bayes\", \"Support Vector Machine\"]\n",
    ")\n",
    "\n",
    "# Feature selection\n",
    "selected_features = st.multiselect(\"Select Features\", iris.feature_names, default=iris.feature_names)\n",
    "\n",
    "if len(selected_features) < 2:\n",
    "    st.warning(\"Please select at least two features.\")\n",
    "    st.stop()\n",
    "\n",
    "X = df[selected_features].values\n",
    "y = df['target'].values\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Model selection\n",
    "model_dict = {\n",
    "    \"Logistic Regression\": LogisticRegression(),\n",
    "    \"Linear Discriminant Analysis\": LinearDiscriminantAnalysis(),\n",
    "    \"K-Nearest Neighbors\": KNeighborsClassifier(),\n",
    "    \"Decision Tree\": DecisionTreeClassifier(),\n",
    "    \"Naive Bayes\": GaussianNB(),\n",
    "    \"Support Vector Machine\": SVC()\n",
    "}\n",
    "\n",
    "model = model_dict[model_name]\n",
    "model.fit(X_train, y_train)\n",
    "preds = model.predict(X_val)\n",
    "\n",
    "st.subheader(\"2. Model Evaluation\")\n",
    "st.write(f\"**Accuracy:** {accuracy_score(y_val, preds):.2f}\")\n",
    "\n",
    "# Show confusion matrix\n",
    "st.write(\"**Confusion Matrix**\")\n",
    "st.write(pd.DataFrame(confusion_matrix(y_val, preds), \n",
    "                      columns=iris.target_names, \n",
    "                      index=iris.target_names))\n",
    "\n",
    "# Show classification report\n",
    "st.write(\"**Classification Report**\")\n",
    "st.text(classification_report(y_val, preds, target_names=iris.target_names))\n",
    "\n",
    "# Visualizations\n",
    "st.subheader(\"3. Pairplot Visualization\")\n",
    "fig = sns.pairplot(df, hue='target_name')\n",
    "st.pyplot(fig)\n",
    "\n",
    "# Box plot\n",
    "st.subheader(\"4. Feature Distributions\")\n",
    "fig2, ax = plt.subplots()\n",
    "df[selected_features].plot(kind='box', subplots=False, ax=ax)\n",
    "st.pyplot(fig2)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
