
### Real Time Prediction of Online Shoppers’ Purchasing Intention using Machine Learning

#### Project Overview
This project aims to predict the purchasing intention of online shoppers in real-time using machine learning techniques. By analyzing various features of user behavior on e-commerce websites, the model will provide insights into the likelihood of a user making a purchase, enabling targeted marketing and improved customer experience.

#### Features
- **Real-Time Data Processing**: The system is designed to handle real-time data, providing immediate predictions as users interact with the website.
- **Machine Learning Models**: Utilizes advanced machine learning models to analyze and predict purchasing intentions.
- **Feature Engineering**: Includes extensive feature engineering to derive meaningful insights from raw data.
- **Visualization and Reporting**: Provides visualizations and reports for better understanding and decision-making.
- **User-Friendly Interface**: A dashboard for easy interaction with the model and visualization of results.

#### Installation

Clone the repository:
```sh
git clone https://github.com/yourusername/online-shoppers-prediction.git
cd online-shoppers-prediction
```

Create a virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scriptsctivate`
```

Install dependencies:
```sh
pip install -r requirements.txt
```

#### Usage

**Data Preparation**:
- Ensure your dataset is in the data directory.
- The dataset should be in CSV format and follow the required schema.

**Train the Model**:
```sh
python main.py
```

**Train Deep Neural Network (DNN) model**:
```sh
python trainDNN.py
```

**Train Long Short-Term Memory (LSTM) model**:
```sh
python trainLSTM.py
```

**Make Predictions**:
```sh
python predict.py --input <input_data_file> --output <output_predictions_file>
```

#### Project Structure
- **data/**: Directory for storing datasets.
- **models/**: Directory for saving trained models.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and model experimentation.
- **scripts/**: Python scripts for data processing, training, and prediction.
- **dashboard.py**: Streamlit application for visualizing predictions and model performance.
- **train.py**: Script for training the machine learning model.
- **predict.py**: Script for making predictions using the trained model.
- **requirements.txt**: List of dependencies required for the project.
- **README.md**: Project documentation.

```
├── EDA.py
├── PredictiveECommerceML.ipynb
├── README.md
├── __init__.py
├── __pycache__
│   ├── EDA.cpython-39.pyc
│   ├── clean_preprocess.cpython-39.pyc
│   └── models.cpython-39.pyc
├── clean_preprocess.py
├── config.py
├── data
│   └── online_shoppers_intention.csv
├── docs
│   ├── AAI-500 Final Team Project Status Update Form Group 6.docx
│   └── Final Team 6 Project Tech Report.docx
├── main.py
├── models
│   ├── Gradient Boosting_model.pkl
│   ├── K-Nearest Neighbors_model.pkl
│   ├── Logistic Regression_model.pkl
│   ├── Random Forest_model.pkl
│   ├── SVM_model.pkl
│   └── XGBoost_model.pkl
├── models.py
├── requirements.txt
├── results
├── trainDNN.py
└── trainLSTM.py
```

#### Contributing
Contributions are welcome! Please read the `CONTRIBUTING.md` file for guidelines on how to get involved.

#### License
This project is licensed under the MIT License - see the `LICENSE` file for details.
