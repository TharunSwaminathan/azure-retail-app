# Basket Analysis

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load datasets
Households5 = pd.read_csv('400_households.csv')
Products = pd.read_csv('400_products.csv')
Transactions3 = pd.read_csv('400_transactions.csv')

# Improved Graph Look and Feel
plt.style.use('ggplot')  # Updated style for a clean look

# Updated Neon Colors
neon_colors = {
    'blue': '#6C63FF',
    'pink': '#F45B69',
    'yellow': '#FFC75F',
    'green': '#61C0BF'
}

# Clean column names
Households5.columns = Households5.columns.str.strip()
Products.columns = Products.columns.str.strip()
Transactions3.columns = Transactions3.columns.str.strip()

# Merge datasets
data = Transactions3.merge(Products, on="PRODUCT_NUM", how="left")
data = data.merge(Households5, on="HSHD_NUM", how="left")

# Basket Analysis Preparation
basket_data = data.groupby(['BASKET_NUM', 'HSHD_NUM'])['COMMODITY'].apply(list).reset_index()

# One-hot encoding
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
basket_matrix = te.fit_transform(basket_data['COMMODITY'])
basket_df = pd.DataFrame(basket_matrix, columns=te.columns_)
basket_df['HSHD_NUM'] = basket_data['HSHD_NUM']
basket_df['BASKET_NUM'] = basket_data['BASKET_NUM']

# Define target product
target_product = 'DAIRY'

# Check if target product is in columns, and dynamically choose another if not
if target_product not in basket_df.columns:
    print(f"Target product '{target_product}' not found.")
    # Dynamically select another product from the columns
    available_products = [col for col in basket_df.columns if col not in ['HSHD_NUM', 'BASKET_NUM']]
    if available_products:
        target_product = available_products[0]  # Choose the first available product
        print(f"Using '{target_product}' as the new target product.")
    else:
        raise ValueError("No valid Products available for analysis!")

# Proceed with feature-target split
X = basket_df.drop(columns=[target_product, 'HSHD_NUM', 'BASKET_NUM'])
y = basket_df[target_product]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save accuracy to a text file
with open('static/model_accuracy.txt', 'w') as f:
    f.write(f"Model Accuracy: {accuracy:.2f}")

# Feature importance
importance = pd.DataFrame({
    'Product': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Feature Importance Bar Chart
plt.figure(figsize=(10, 6))
plt.barh(
    importance['Product'][:10], importance['Importance'][:10],
    color=sns.color_palette("cool", n_colors=10),
    edgecolor='white'
)
plt.xlabel('Importance', fontsize=14, color=neon_colors['green'])
plt.ylabel('Product', fontsize=14, color=neon_colors['green'])
plt.title(f'Top Products Purchased with {target_product}', fontsize=16, color=neon_colors['blue'])
plt.gca().invert_yaxis()
plt.xticks(fontsize=12, color=neon_colors['pink'])
plt.yticks(fontsize=12, color=neon_colors['pink'])
plt.grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.savefig('static/feature_importance_updated.png', dpi=300, bbox_inches="tight")
print("Updated Feature Importance Bar Chart saved.")

# Updated Network Graph
co_occurrence = basket_df.drop(columns=['HSHD_NUM', 'BASKET_NUM']).T.dot(
    basket_df.drop(columns=['HSHD_NUM', 'BASKET_NUM'])
)
np.fill_diagonal(co_occurrence.values, 0)

G = nx.Graph()
threshold = 0.05
for product_a, product_b in co_occurrence.stack().index:
    weight = co_occurrence.loc[product_a, product_b]
    if weight > threshold:
        G.add_edge(product_a, product_b, weight=weight)

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.3, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=700, node_color=sns.color_palette("cool", n_colors=G.number_of_nodes()))
nx.draw_networkx_edges(G, pos, edgelist=G.edges(data=True), width=1.2, alpha=0.6, edge_color=neon_colors['yellow'])
nx.draw_networkx_labels(G, pos, font_size=10, font_color=neon_colors['green'], font_weight="bold")
plt.title("Network Graph of Product Associations", fontsize=16, color=neon_colors['blue'])
plt.tight_layout()
plt.savefig('static/product_network_updated.png', dpi=300, bbox_inches="tight")
print("Updated Network Graph saved.")

# Updated Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    co_occurrence, cmap="mako", linewidths=0.8, annot=False, square=True,
    linecolor=neon_colors['green']
)
plt.title('Co-Purchase Frequency Heatmap', fontsize=16, color=neon_colors['yellow'])
plt.tight_layout()
plt.savefig('static/heatmap_updated.png', dpi=300, bbox_inches="tight")
print("Updated Heatmap saved.")
