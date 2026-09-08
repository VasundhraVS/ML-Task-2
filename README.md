# Customer Segmentation Using RFM Analysis and K-Means Clustering

## 📌 Project Overview

This project performs **customer segmentation** using **RFM (Recency, Frequency, Monetary) analysis** and **K-Means clustering**.

The objective is to analyze customer purchasing behavior and group customers into meaningful segments based on their transaction history.

The project uses Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

---

## 🎯 Objectives

* Clean and preprocess the sales transaction dataset.
* Calculate important customer-level RFM metrics.
* Scale the RFM features for machine learning.
* Apply K-Means clustering to segment customers.
* Use the Elbow Method to evaluate different numbers of clusters.
* Evaluate clustering performance using:

  * Silhouette Score
  * Davies-Bouldin Index
* Visualize and summarize customer segments.

---

## 📊 Dataset

The project uses a sales transaction dataset containing information such as:

* **Invoice** – Invoice/transaction identifier
* **InvoiceDate** – Date of the transaction
* **Customer ID** – Unique customer identifier
* **Quantity** – Number of products purchased
* **Price** – Price per product

A new feature called **TotalAmount** is calculated from:

```text
TotalAmount = Quantity × Price
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Visualization
* **Scikit-learn** – Machine learning

  * StandardScaler
  * KMeans
  * Silhouette Score
  * Davies-Bouldin Index
* **Google Colab / Jupyter Notebook**

---

## 🔄 Project Workflow

```text
Sales Dataset
      ↓
Data Loading
      ↓
Data Inspection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
RFM Calculation
      ↓
Feature Scaling
      ↓
K-Means Clustering
      ↓
Cluster Evaluation
      ↓
Visualization
      ↓
Customer Segmentation
```

---

## 🧹 Data Cleaning and Preprocessing

The following preprocessing steps were performed:

### 1. Load the Dataset

The sales dataset is loaded using Pandas:

```python
df = pd.read_csv("sales2.xlsx - Sheet1.csv")
```

### 2. Inspect the Dataset

The dataset is explored using:

```python
df.head()
df.info()
df.tail()
df.isnull()
df.describe()
df.shape
```

### 3. Remove Missing Customer IDs

Transactions without a customer identifier are removed:

```python
df = df.dropna(subset=["Customer ID"])
```

### 4. Remove Cancelled Invoices

Cancelled invoices are excluded from the analysis.

```python
df = df[~df["InvoiceDate"].astype(str).str.startswith("C")]
```

### 5. Remove Invalid Transactions

Transactions with zero or negative quantity/price are removed:

```python
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
```

### 6. Remove Duplicate Records

Duplicate transactions are removed:

```python
df = df.drop_duplicates()
```

---

## 💰 Feature Engineering

A new **TotalAmount** column is created:

```python
df["TotalAmount"] = df["Quantity"] * df["Price"]
```

This represents the total monetary value of each transaction.

---

# 📈 RFM Analysis

RFM analysis measures customer behavior using three important metrics.

### Recency

Measures **how recently a customer made a purchase**.

```text
Lower Recency = More recent purchase
```

### Frequency

Measures **how frequently a customer makes purchases**.

```text
Higher Frequency = More purchases
```

### Monetary

Measures **how much money a customer has spent**.

```text
Higher Monetary = Higher spending
```

The RFM table is created using customer-level aggregation:

```python
rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate",
             lambda x: (reference_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalAmount", "sum")
)
```

The reference date is calculated as one day after the latest transaction date.

---

## ⚖️ Feature Scaling

Since Recency, Frequency, and Monetary have different scales, **StandardScaler** is used before clustering.

```python
from sklearn.preprocessing import StandardScaler

features = ["Recency", "Frequency", "Monetary"]

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm[features])
```

Scaling prevents features with larger numerical values from dominating the clustering algorithm.

---

# 🤖 K-Means Clustering

K-Means clustering is used to group customers with similar purchasing behavior.

Different values of **K**, from 2 to 10, are tested.

```python
for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(rfm_scaled)
    inertia.append(model.inertia_)
```

---

## 📉 Elbow Method

The **Elbow Method** is used to evaluate the inertia values for different numbers of clusters.

```python
plt.plot(range(2, 11), inertia, marker="o")
```

The notebook selects **2 clusters** for the final K-Means model.

```python
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
```

---

# 📏 Cluster Evaluation

Two evaluation metrics are used to assess the quality of the clustering.

## 1. Silhouette Score

The Silhouette Score measures how well customers fit within their assigned clusters.

```python
from sklearn.metrics import silhouette_score

silhouette = silhouette_score(
    rfm_scaled,
    rfm["Cluster"]
)

print("Silhouette Score:", silhouette)
```

A higher Silhouette Score generally indicates better-defined clusters.

---

## 2. Davies-Bouldin Index

The Davies-Bouldin Index evaluates the similarity between clusters.

```python
from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(
    rfm_scaled,
    rfm["Cluster"]
)

print("Davies-Bouldin Index:", db_index)
```

A lower Davies-Bouldin Index generally indicates better cluster separation.

---

# 📊 Customer Cluster Visualization

The customer clusters are visualized using Frequency and Monetary values.

```python
plt.figure(figsize=(8, 6))

plt.scatter(
    rfm["Frequency"],
    rfm["Monetary"],
    c=rfm["Cluster"]
)

plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.title("Customer Clusters")
plt.show()
```

This visualization helps understand differences in customer purchasing behavior.

---

# 📋 Cluster Summary

The average RFM values for each customer cluster are calculated:

```python
cluster_summary = rfm.groupby("Cluster")[
    ["Recency", "Frequency", "Monetary"]
].mean()

print(cluster_summary)
```

This summary can be used to understand the characteristics of each customer segment.

For example:

| RFM Metric     | Interpretation                      |
| -------------- | ----------------------------------- |
| Low Recency    | Customer purchased recently         |
| High Recency   | Customer has not purchased recently |
| High Frequency | Customer purchases frequently       |
| High Monetary  | Customer spends more                |
| Low Monetary   | Customer spends less                |

---

## 💡 Business Applications

The customer segments generated through this analysis can support:

* Customer retention strategies
* Personalized marketing campaigns
* Customer loyalty programs
* Targeted promotions
* Identifying high-value customers
* Identifying customers who may need re-engagement
* Improving customer relationship management

---

## 📁 Project Structure

```text
Customer-Segmentation/
│
├── Machine_Learning_Task_2.ipynb
├── sales2.xlsx - Sheet1.csv
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the notebook

Open:

```text
Machine_Learning_Task_2.ipynb
```

using **Google Colab** or **Jupyter Notebook**.

### 3. Upload the dataset

Make sure the sales CSV file is available in the notebook environment.

### 4. Install required libraries

If necessary:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 5. Run the notebook

Execute the cells sequentially to perform data cleaning, RFM analysis, clustering, evaluation, and visualization.

---

## 📌 Key Takeaways

This project demonstrates an end-to-end customer segmentation workflow using:

**Data Cleaning → RFM Analysis → Feature Scaling → K-Means Clustering → Cluster Evaluation → Visualization**

The analysis provides a machine-learning-based approach for understanding customer purchasing behavior and creating meaningful customer segments.

---

## 👩‍💻 Author

**Vasundhra**

BCA Student | Aspiring Data Analyst

### Skills Demonstrated

`Python` `Pandas` `NumPy` `Data Cleaning` `RFM Analysis` `K-Means` `Machine Learning` `Data Visualization` `Scikit-learn`
