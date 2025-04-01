# Overview
Banks and credit card agencies constantly face challenges like fraud, fraudulent transactions, and misrepresentation. 
This project evaluates various classification models for detecting fraudulent credit card transactions. Using the Kaggle Credit Card Fraud Detection dataset, we preprocess the data, handle class imbalance, and implement machine learning models to improve fraud detection accuracy.

![image](https://github.com/user-attachments/assets/92caca6b-de54-4389-ab66-74b8eece4eb8)

## Dataset
The dataset is sourced from Kaggle:
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **284,807** transactions
- **30 anonymized features**
- **Class column:** 0 (non-fraud) and 1 (fraud)
- Highly imbalanced (fraud accounts for ~0.2% of data)
  
![image](https://github.com/user-attachments/assets/aa0b7e7e-d960-43de-a63f-42ff3198f0f2)

## Technologies Used
- **Python**
- **Pandas, NumPy** (Data handling)
- **Matplotlib, Seaborn** (Visualization)
- **Scikit-learn** (Machine Learning)
- **TensorFlow/Keras** (Neural Networks)

## Setup Instructions
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place `creditcard.csv` in the project directory.
4. Run the Colab Notebook to train and evaluate models.

## Model Implementation
The following models were implemented and evaluated:
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Neural Networks (Keras Sequential Model)**

### Performance Metrics
- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**

## **Confusion Matrices**
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/f0d34e1d-21cc-45b9-b923-415cc58cbf6e" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/ee02ee72-0eea-4fc4-b620-cb6c7124683d" width="400"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/28ca6b30-7755-46d3-97eb-4402e8465290" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/5218c771-cd62-4aa7-b6b0-c4b8c1768dba" width="400"></td>
  </tr>
</table>


## **Model Comparison and Performance Analysis**

Below is the comparison of four models — Logistic Regression, Random Forest, Gradient Boosting, and Shallow Neural Network — based on their performance in predicting fraudulent transactions.

#### **Performance Metrics:**

| **Model**                      | **Accuracy** | **Precision (Not Fraud)** | **Recall (Not Fraud)** | **F1-Score (Not Fraud)** | **Precision (Fraud)** | **Recall (Fraud)** | **F1-Score (Fraud)** |
|---------------------------------|--------------|---------------------------|------------------------|--------------------------|-----------------------|--------------------|----------------------|
| **Logistic Regression**         | 1.00         | 1.00                      | 1.00                   | 1.00                     | 0.94                  | 0.63               | 0.76                 |
| **Random Forest**               | 1.00         | 1.00                      | 1.00                   | 1.00                     | 0.90                  | 0.55               | 0.68                 |
| **Gradient Boosting**           | 1.00         | 1.00                      | 1.00                   | 1.00                     | 0.85                  | 0.67               | 0.75                 |
| **Shallow Neural Network**      | 0.9995       | 1.00                      | 1.00                   | 1.00                     | 0.85                  | 0.67               | 0.75                 |

---

### **Analysis**

1. **Logistic Regression**:
   - **Strengths**: Achieved perfect performance in predicting non-fraudulent transactions with perfect precision and recall.
   - **Weaknesses**: Struggled to predict fraudulent transactions, with a relatively low recall and F1-score for the "Fraud" class.

2. **Random Forest**:
   - **Strengths**: Perfect performance in predicting non-fraudulent transactions. It demonstrated a good balance for fraud prediction with reasonable precision and recall.
   - **Weaknesses**: Its recall for fraudulent transactions was lower than expected, indicating some fraud cases were missed.

3. **Gradient Boosting**:
   - **Strengths**: High performance in predicting non-fraudulent transactions. It performed reasonably well in predicting fraud, with a higher recall than Random Forest.
   - **Weaknesses**: While better than Random Forest, the precision for fraud detection was still moderate.

4. **Shallow Neural Network**:
   - **Strengths**: Excellent performance in predicting non-fraudulent transactions, with good precision and recall for fraud predictions.
   - **Weaknesses**: Similar to Gradient Boosting, its recall for fraud predictions is relatively lower, but its precision is still good.

### **Best Model**:
- **Logistic Regression** provided the best overall accuracy and performance for predicting non-fraudulent transactions, but for fraud detection, **Gradient Boosting** and **Shallow Neural Network** performed slightly better, offering better recall and F1-scores for the "Fraud" class. However, the **Random Forest** model performed similarly to Gradient Boosting and Neural Network, making it a good alternative based on its computational efficiency.

## Future Improvements
- Experiment with **Anomaly Detection** techniques.
- SMOTE (Synthetic Minority Over-sampling Technique) to improve model performance.
- Hypermetric tuning for model effectiveness. 
- Apply **Graph Neural Networks (GNNs)** for improved fraud detection.
- Deploy the model using **Flask or FastAPI** for real-time predictions.

## Contributing
Feel free to fork this repository and submit a pull request with improvements!

## License
This project is licensed under the **MIT License**.
