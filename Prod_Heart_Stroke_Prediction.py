import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy import stats

warnings.filterwarnings('ignore')

def send_email(subject, body, sender_email, receiver_email, smtp_server, smtp_port, smtp_user, smtp_password):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()

        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def classification_eval(y_test, y_pred):
    return {
        'accuracy': np.round(accuracy_score(y_test, y_pred), 3),
        'precision': np.round(precision_score(y_test, y_pred), 3),
        'recall': np.round(recall_score(y_test, y_pred), 3),
        'f1_score': np.round(f1_score(y_test, y_pred), 3),
        'roc_auc': np.round(roc_auc_score(y_test, y_pred), 3),
        'null_accuracy': round(max(y_test.mean(), 1 - y_test.mean()), 2)
    }

def stroke_prediction_pipeline(filepath, sender_email, receiver_email, smtp_server, smtp_port, smtp_user, smtp_password):
    try:
        # Data Loading and Initial Review
        df = pd.read_csv(filepath)
        print("The shape of the data:\n", df.shape)
        print("\nThe first 5 rows are:\n", df.head())
        print("\nThe last 5 rows are:\n", df.tail())
        print("\nThe column names are:\n", df.columns)
        
        # Data Cleaning and Preprocessing
        df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
        df = df.drop(columns='id')
        df = df[df.gender != "Other"]
        
        # Outlier removal for BMI
        df["bmi"] = pd.to_numeric(df["bmi"])
        df["bmi"] = df["bmi"].apply(lambda x: 48 if x > 48 else x)
        
        # Outlier detection using IsolationForest
        from sklearn.ensemble import IsolationForest
        isolation = IsolationForest(n_estimators=1000, contamination=0.03)
        out = pd.Series(isolation.fit_predict(df[['bmi', 'avg_glucose_level']]), name='outliers')
        df = pd.concat([out.reset_index(), df.reset_index()], axis=1, ignore_index=False).drop(columns='index')
        df = df[df['outliers'] == 1]
        
        # Standardize numerical features
        scaler = StandardScaler()
        df[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(df[['age', 'avg_glucose_level', 'bmi']])
        
        # Label encoding for categorical features
        encoder = LabelEncoder()
        df['gender'] = encoder.fit_transform(df['gender'])
        df['ever_married'] = encoder.fit_transform(df['ever_married'])
        df['work_type'] = encoder.fit_transform(df['work_type'])
        df['Residence_type'] = encoder.fit_transform(df['Residence_type'])
        df['smoking_status'] = encoder.fit_transform(df['smoking_status'])
        
        df_encoded = df
        
        # Modeling
        X = df_encoded.drop(columns=['stroke', 'outliers'])
        y = df_encoded.stroke
        
        # Balancing the dataset using SMOTE
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)
        
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
        
        # RandomForestClassifier
        randomf = RandomForestClassifier()
        randomf.fit(X_train, y_train)
        y_pred = randomf.predict(X_test)
        
        # Evaluation
        eval_metrics = classification_eval(y_test, y_pred)
        print(eval_metrics)
        
        # Confusion Matrix
        plot_confusion_matrix(randomf, X_test, y_test, cmap="BuPu")
        plt.title("Confusion Matrix")
        plt.show()
        
        # Send email notification with results
        subject = "Stroke Prediction Pipeline Results"
        body = (f"Accuracy: {eval_metrics['accuracy']}\n"
                f"Precision: {eval_metrics['precision']}\n"
                f"Recall: {eval_metrics['recall']}\n"
                f"F1 Score: {eval_metrics['f1_score']}\n"
                f"ROC AUC: {eval_metrics['roc_auc']}\n"
                f"Null Accuracy: {eval_metrics['null_accuracy']}")
        send_email(subject, body, sender_email, receiver_email, smtp_server, smtp_port, smtp_user, smtp_password)

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")
        # Send email notification with error
        subject = "Stroke Prediction Pipeline Error"
        body = f"An error occurred: {e}"
        send_email(subject, body, sender_email, receiver_email, smtp_server, smtp_port, smtp_user, smtp_password)

# Email configuration
sender_email = "prod_run@dummy_email.com"
receiver_email = "redacted"
smtp_server = "redacted"
smtp_port = 25
smtp_user = "redacted"
smtp_password = "redacted"

# Run the pipeline
stroke_prediction_pipeline("healthcare-dataset-stroke-data.csv", sender_email, receiver_email, smtp_server, smtp_port, smtp_user, smtp_password)
