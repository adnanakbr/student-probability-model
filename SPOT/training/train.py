"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os


# Split the dataframe into test and train data
def split_data(df):
    X = df.drop('target_rP', axis=1).values
    y = df['target_rP'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data


# Train the model, return the model
def train_model(data, args):
    rf_model = RandomForestClassifier(**args)
    rf_model.fit(data["train"]["X"], data["train"]["y"])
    return rf_model


# Evaluate the metrics for the model
def get_model_metrics(rf_model, data):
    preds = rf_model.predict(data["test"]["X"])
    accuracy = balanced_accuracy_score(preds, data["test"]["y"])
    metrics = {"accuracy": accuracy}
    return metrics


def main():
    print("Running train.py")

    # Define training parameters
    args = {"max_depth": 8}

    # Load the training data as dataframe
    data_dir = "/home/adnan/OU/azure-devops/mlops/student-probability-model/data"
    data_file = os.path.join(data_dir, 'student_data.csv')
    train_df = pd.read_csv(data_file)
    #pre-processing - convert categorical columns into label encoding
    category_columns = train_df.select_dtypes(include=['object']).columns
    for column in category_columns:
        train_df[column] = train_df[column].astype('category')
        train_df[column] = train_df[column].cat.codes

    train_df = train_df.fillna(train_df.mean())

    data = split_data(train_df)

    # Train the model
    model = train_model(data, args)

    # Log the metrics for the model
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
