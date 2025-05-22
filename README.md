import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

sns.set(style="whitegrid")


# ---

import os
os.getcwd()


# ---

with zipfile.ZipFile("netflix_titles.csv.zip", "r") as zip_ref:
    zip_ref.extractall(".")

# Load CSV
df = pd.read_csv("netflix_titles.csv")

# Preview
df.head()

# ---

import pandas as pd

# Load dataset (assuming CSV is in the same folder)
df = pd.read_csv("netflix_titles.csv")

# Handle missing values safely
columns_to_fill = ['director', 'cast', 'country', 'rating', 'duration']
for col in columns_to_fill:
    df[col] = df[col].fillna('Unknown')


# ---

# Convert 'date_added' to datetime format
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Extract year and month from 'date_added'
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month_name()

# Split genres into list
df['genres'] = df['listed_in'].str.split(', ')

# Extract numerical part from 'duration'
df['duration_int'] = df['duration'].str.extract(r'(\d+)').astype('float')

# Extract type part from 'duration'
df['duration_type'] = df['duration'].str.extract(r'([a-zA-Z]+)')


# ---

# Check for nulls in date
print("Missing values in 'date_added':", df['date_added'].isna().sum())

# Check unusual values in 'duration_type'
print("Unique duration types:", df['duration_type'].unique())


# ---

# Standardize 'type', 'rating', and 'country' fields
df['type'] = df['type'].str.strip().str.title()
df['rating'] = df['rating'].str.strip().str.upper()
df['country'] = df['country'].str.strip()


# ---

# Overview of data
df.info()


# ---

# Summary statistics
df.describe()


# ---

# Count of Movies vs TV Shows
df['type'].value_counts()


# ---

# Top 10 content-producing countries
df['country'].value_counts().head(10)


# ---

# Top ratings
df['rating'].value_counts().head(10)


# ---

# Content added per year
df['year_added'].value_counts().sort_index()


# ---

# Top 5 directors
df['director'].value_counts().head(5)


# ---

# Number of titles added per year
content_per_year = df['year_added'].value_counts().sort_index()
print(content_per_year)


# ---

# Group by year and type
type_trend = df.groupby(['year_added', 'type']).size().unstack()
print(type_trend)


# ---

import matplotlib.pyplot as plt
import seaborn as sns

# Optional: make plots look better
sns.set(style='darkgrid')


# ---

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='type', palette='pastel')
plt.title('Content Type Distribution')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()


# ---

plt.figure(figsize=(10,5))
df['year_added'].value_counts().sort_index().plot(kind='bar', color='blue')
plt.title('Content Added Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.show()


# ---

plt.figure(figsize=(10,5))
df['country'].value_counts().head(10).plot(kind='bar', color='green')
plt.title('Top 10 Content-Producing Countries')
plt.xlabel('Country')
plt.ylabel('Number of Titles')
plt.xticks(rotation=45)
plt.show()


# ---

from collections import Counter

genre_counter = Counter()
df['genres'].dropna().apply(lambda x: genre_counter.update(x))
top_genres = dict(genre_counter.most_common(10))

plt.figure(figsize=(10,5))
sns.barplot(x=list(top_genres.keys()), y=list(top_genres.values()), palette='viridis')
plt.title('Top 10 Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# ---

# Pie chart of content type
plt.figure(figsize=(6,6))
df['type'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
plt.title('Distribution of Content Type')
plt.ylabel('')
plt.show()


# ---

# Count of content added by year and month
heatmap_data = df.pivot_table(index='month_added', columns='year_added', values='title', aggfunc='count')

# Order months properly
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
heatmap_data = heatmap_data.reindex(month_order)

plt.figure(figsize=(14,6))
sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=0.5, linecolor='white')
plt.title('Content Additions by Month and Year')
plt.xlabel('Year')
plt.ylabel('Month')
plt.show()


# ---

# Boxplot to visualize outliers in duration
plt.figure(figsize=(8, 4))
sns.boxplot(data=df[df['type'] == 'Movie'], x='duration_int', color='salmon')
plt.title('Outliers in Movie Duration (minutes)')
plt.xlabel('Duration (minutes)')
plt.show()


# ---

# Apply log transformation to movie durations (if you want to normalize the distribution)
import numpy as np

df['log_duration'] = df['duration_int'].apply(lambda x: np.log(x + 1))  # +1 to avoid log(0)


# ---

plt.figure(figsize=(8, 4))
sns.histplot(df[df['type'] == 'Movie']['log_duration'], bins=30, color='skyblue')
plt.title('Log-Transformed Movie Durations')
plt.xlabel('log(Duration + 1)')
plt.show()
