from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# Jinja2 for templates
templates = Jinja2Templates(directory="templates")

# Load dataset
df = pd.read_csv("./data/df_merged.csv")

# Ensure 'user_id' is of type str
df['user_id'] = df['user_id'].astype(str)
df['user_id'] = df['user_id'].str.strip()

# Create a combined text column for embeddings
df["product_info"] = (
    df["Product Name"].astype(str) + " " +
    df["Category"].astype(str) + " " +
    df["brand"].astype(str) + " " +
    df["About Product"].astype(str) + " " +
    df["Technical Details"].astype(str)
)

# Load precomputed embeddings (ensure this is done only once)
embeddings_file = "./data/embeddings.pkl"
embeddings_dict = {}

# Load embeddings only when requested for the first time
if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        embeddings_dict = pickle.load(f)
    print("Embeddings loaded successfully.")
    print(f"Embeddings dictionary contains {len(embeddings_dict)} entries.")
else:
    raise FileNotFoundError("Embeddings file not found.")

# Map embeddings to the dataframe based on 'product_id'
df["embeddings"] = df["product_info"].map(embeddings_dict)

# Getting the most purchased products (for fallback recommendations)
popular_products = df["product_id"].value_counts().index[:10].tolist()
popular_recommendations = df[df["product_id"].isin(popular_products)][["product_id", "Product Name", "price"]].drop_duplicates()

# Ensure there are valid embeddings (non-null values)
valid_embeddings = df["embeddings"].dropna()

# If there are no valid embeddings:
if valid_embeddings.empty:
    raise ValueError("No valid embeddings found in the dataset.")

# Create the embedding_matrix if valid embeddings exist
embedding_matrix = np.stack(valid_embeddings.values)

# Recommendation function
def recommend_products(user_id, df, popular_recommendations, top_n=10):
    # Get the user's purchase history and remove products with missing embeddings
    user_products = df[df["user_id"] == user_id].dropna(subset=["embeddings"])
    print(f"User {user_id} has purchased {len(user_products)} products.")

    # If the user has no purchase history, return diverse popular recommendations
    if user_products.empty:
        available_popular = popular_recommendations.drop_duplicates(subset=["Product Name"])
        return available_popular.sample(n=min(top_n, len(available_popular)), replace=False).to_dict(orient="records")

    # Extract user embeddings from their purchased products
    user_embeddings = np.stack(user_products["embeddings"].values)

    # Get valid embeddings (products with embeddings)
    valid_embeddings = df.dropna(subset=["embeddings"])
    if valid_embeddings.empty:
        available_popular = popular_recommendations.drop_duplicates(subset=["Product Name"])
        return available_popular.sample(n=min(top_n, len(available_popular)), replace=False).to_dict(orient="records")

    embedding_matrix = np.stack(valid_embeddings["embeddings"].values)

    # Compute cosine similarity between user embeddings and all product embeddings
    similarities = cosine_similarity(user_embeddings, embedding_matrix).mean(axis=0)

    # Assign similarity scores to each product
    valid_embeddings = valid_embeddings.copy()
    valid_embeddings["similarity"] = similarities  

    # Remove already purchased products
    purchased_products = set(user_products["product_id"].values)
    df_filtered = valid_embeddings[~valid_embeddings["product_id"].isin(purchased_products)]

    # Sort by similarity (highest first)
    df_filtered = df_filtered.sort_values(by="similarity", ascending=False)

    # Select top products while ensuring variety
    recommended = df_filtered.groupby("Product Name").head(3).reset_index(drop=True).head(top_n)
    print(recommended[["Product Name", "Image"]].head())

    # If there are not enough recommendations, fill up with popular products
    if len(recommended) < top_n:
        remaining = top_n - len(recommended)
        available_popular = popular_recommendations.drop_duplicates(subset=["Product Name"])
        additional_products = available_popular.sample(n=min(remaining, len(available_popular)), replace=False)
        recommended = pd.concat([recommended, additional_products])
        print(f"After adding popular products: {len(recommended)} total recommendations.")

    # Ensure we are using only the first image URL (split by '|')
    recommended["Image"] = recommended["Image"].apply(lambda x: x.split('|')[0] if isinstance(x, str) else "")
    


    # Filter to ensure there are no null values in the required columns
    recommended = recommended.dropna(subset=["Image", "Product Url"])

    # Return the top recommended products with relevant details
    return recommended[["product_id", "Product Name", "price", "Selling Price", "Image", "Product Url", "similarity"]].head(top_n).to_dict(orient="records")

# API endpoint
@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, request: Request):
    user_id = str(user_id)  # Ensure user_id is a string

    recommendations = recommend_products(user_id, df, popular_recommendations=popular_recommendations)
    return templates.TemplateResponse("recommendations.html", {"request": request, "recommendations": recommendations})
