
# Prediction Model: Chicago Taxi Fares 

This project implements a **Simple Linear Regression** model using **TensorFlow** and **Keras** to predict taxi fares in Chicago. It focuses on identifying the relationship between travel distance (`TRIP_MILES`) and the total cost of the trip (`FARE`).

##  Features

* **Data Exploration**: Extensive analysis of the Chicago Taxi dataset, including statistical descriptions and correlation matrices.
* **Visualization**: Interactive data plotting using `Plotly` and `Seaborn` to identify trends and outliers.
* **Deep Learning Framework**: Built using `TensorFlow 2.18` and `Keras 3.8`.
* **Predictive Analytics**: A trained linear model that predicts the monetary cost of a taxi ride based on mileage.

##  Technical Architecture

The model architecture is a minimal **Sequential Neural Network**:

1. **Input Layer**: Accepts a single feature (`TRIP_MILES`).
2. **Dense Layer**: A single neuron with a linear activation function (representing ).
3. **Optimizer**: Stochastic Gradient Descent (SGD) or Adam (standard for regression tasks).
4. **Loss Function**: Mean Squared Error (MSE), used to calculate the variance between predicted and actual fares.

##  Setup & Installation

### Prerequisites

* Python 3.10+
* Jupyter Notebook or Google Colab

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/xenithrider/prediction_model.git
cd prediction_model

```


2. **Install dependencies**:
```bash
pip install keras~=3.8.0 matplotlib~=3.10.0 numpy~=2.0.0 pandas~=2.2.0 tensorflow~=2.18.0 plotly seaborn

```



### Usage

Run the `linear_regression.ipynb` notebook to:

1. Load the live Chicago Taxi dataset from Google's MLCC repository.
2. Perform Exploratory Data Analysis (EDA).
3. Train the linear regression model.
4. Evaluate the model performance against actual fare data.

---

##  Dataflow Explanation

The dataflow of the **Prediction Model** follows a linear pipeline from raw data ingestion to predictive output:

1. **Ingestion**: The script fetches the `chicago_taxi_train.csv` dataset directly from a remote Google URL into a Pandas DataFrame.
2. **Preprocessing & EDA**:
* The data is cleaned and inspected using `df.describe()` and `df.info()`.
* A **Correlation Matrix** is generated to confirm that `TRIP_MILES` is a strong predictor for `FARE` (typically showing a high positive correlation).
* Visual inspection through **Pair Plots** helps identify linear trends and potential data noise.


3. **Feature Selection**: The flow narrows down from 18 initial columns (like `TRIP_START_HOUR`, `PAYMENT_TYPE`, etc.) to a specific **Feature** (`TRIP_MILES`) and a **Label** (`FARE`).
4. **Model Training**:
* The data is passed into the Keras `Sequential` model.
* During each epoch, the model adjusts the "weights" (the slope) and "bias" (the intercept) to minimize the error between its prediction and the real fare.


5. **Inference**: Once trained, new mileage values can be fed into the model to receive a predicted fare amount in USD.
