# Step-by-Step Guide to Implementing H3 Indexing for Geospatial Data

## Introduction

H3 is a hierarchical geospatial indexing system developed by Uber that divides the Earth's surface into hexagonal cells. It's particularly useful for location-based analytics, efficient spatial queries, and visualization of geospatial data. This guide will walk you through implementing H3 indexing for your restaurant dataset.

## Prerequisites

Before we begin, make sure you have the following installed:

```bash
pip install pandas h3 folium matplotlib geopandas shapely
```

## Step 1: Understanding H3 Indexing

H3 uses a hierarchical system with different resolutions:
- Lower resolution (0-5): Larger hexagons, good for country/regional analysis
- Medium resolution (6-9): Medium-sized hexagons, good for city/neighborhood analysis
- Higher resolution (10-15): Smaller hexagons, good for street-level analysis

Each location (lat, long) corresponds to a unique H3 index at each resolution level. The H3 index is a 64-bit integer represented as a hexadecimal string.

## Step 2: Loading and Preparing the Data

First, let's load our restaurant data:

```python
import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Check for missing values in latitude/longitude
    missing_coords = df[df['latitude'].isna() | df['longitude'].isna()]
    if not missing_coords.empty:
        print(f"Warning: Found {len(missing_coords)} records with missing coordinates.")
        # Drop rows with missing coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
    
    print(f"Loaded {len(df)} records with valid coordinates.")
    
    # Display basic statistics
    print(f"Latitude range: {df['latitude'].min()} to {df['latitude'].max()}")
    print(f"Longitude range: {df['longitude'].min()} to {df['longitude'].max()}")
    
    return df

# Load the restaurant data
df = load_data('synthesized_restaurants.csv')
```

## Step 3: Implementing H3 Indexing

Now, let's add H3 indices to our data at different resolution levels:

```python
import h3

def add_h3_indices(df, resolutions=[7, 8, 9, 10]):
    """Add H3 indices at specified resolution levels to the dataframe"""
    print(f"Generating H3 indices at resolutions: {resolutions}...")
    
    # Create a copy of the dataframe
    df_h3 = df.copy()
    
    # Add H3 index columns for each resolution
    for res in resolutions:
        col_name = f'h3_index_{res}'
        df_h3[col_name] = df_h3.apply(
            lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], res), 
            axis=1
        )
        
        # Count unique hexagons at this resolution
        unique_hexagons = df_h3[col_name].nunique()
        print(f"Resolution {res}: {unique_hexagons} unique hexagons")
    
    return df_h3

# Add H3 indices at different resolutions
resolutions = [6, 9, 12]  # Low, medium, and high resolutions
df_h3 = add_h3_indices(df, resolutions)

# Save the indexed data for future use
df_h3.to_csv('restaurants_h3_indexed.csv', index=False)
```

## Step 4: Analyzing Data Distribution Across H3 Cells

Let's analyze how restaurants are distributed across H3 cells:

```python
def analyze_h3_distribution(df_h3, resolution=9):
    """Analyze the distribution of restaurants across H3 cells"""
    h3_col = f'h3_index_{resolution}'
    
    # Count restaurants per H3 cell
    h3_counts = df_h3[h3_col].value_counts()
    
    print(f"Analyzing restaurant distribution at resolution {resolution}:")
    print(f"Total unique H3 cells: {len(h3_counts)}")
    print(f"Maximum restaurants in a single cell: {h3_counts.max()}")
    print(f"Average restaurants per occupied cell: {h3_counts.mean():.2f}")
    
    # Define density categories
    density_categories = {
        'Very Low (1)': 1,
        'Low (2-5)': 2,
        'Medium (6-20)': 6,
        'High (21-100)': 21,
        'Very High (100+)': 100
    }
    
    # Categorize cells by restaurant density
    density_counts = {}
    prev_threshold = 0
    for category, threshold in density_categories.items():
        count = ((h3_counts >= prev_threshold) & (h3_counts < threshold)).sum()
        if category == list(density_categories.keys())[-1]:  # For the last category (100+)
            count = (h3_counts >= threshold).sum()
        density_counts[category] = count
        prev_threshold = threshold
    
    print("\nH3 cell density distribution:")
    for category, count in density_counts.items():
        percentage = (count / len(h3_counts)) * 100
        print(f"{category}: {count} cells ({percentage:.1f}%)")
    
    return h3_counts

# Analyze distribution at resolution 9 (medium granularity)
h3_counts = analyze_h3_distribution(df_h3, resolution=9)
```

## Step 5: Visualizing H3 Hexagons on a Map

Let's visualize the H3 hexagons on a map to better understand the spatial distribution:

```python
import folium
from folium.plugins import HeatMap
import numpy as np

def visualize_h3_hexagons(df_h3, resolution=9, max_hexagons=1000):
    """Create a folium map visualizing H3 hexagons with restaurant density"""
    h3_col = f'h3_index_{resolution}'
    
    # Get the most populated hexagons for visualization
    h3_counts = df_h3[h3_col].value_counts()
    
    # Limit to max_hexagons to avoid browser performance issues
    if len(h3_counts) > max_hexagons:
        print(f"Limiting visualization to the {max_hexagons} most populated hexagons.")
        h3_counts = h3_counts.head(max_hexagons)
    
    # Calculate the center point for the map
    center_lat = df_h3['latitude'].median()
    center_lng = df_h3['longitude'].median()
    
    # Create a map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=4)
    
    # Add hexagons to the map
    for h3_index, count in h3_counts.items():
        # Get hexagon boundary
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        
        # Calculate color based on restaurant count
        color = get_color_for_count(count, h3_counts.max())
        
        # Create a polygon for each hexagon
        folium.Polygon(
            locations=boundary,
            tooltip=f"H3 Index: {h3_index}<br>Restaurant Count: {count}",
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            color='black',
            weight=1
        ).add_to(m)
    
    # Add a heatmap layer
    heat_data = [[row['latitude'], row['longitude']] for _, row in df_h3.iterrows()]
    HeatMap(heat_data, radius=10).add_to(m)
    
    # Save the map
    output_file = f"restaurant_h3_map_res{resolution}.html"
    m.save(output_file)
    print(f"Map saved to {output_file}")
    
    return output_file

def get_color_for_count(count, max_count):
    """Generate a color on a red-yellow-green scale based on count"""
    # Scale from 0 to 1
    ratio = count / max_count
    
    # Create a color scale
    if ratio < 0.1:
        return 'darkgreen'
    elif ratio < 0.3:
        return 'green'
    elif ratio < 0.5:
        return 'lightgreen'
    elif ratio < 0.7:
        return 'yellow'
    elif ratio < 0.9:
        return 'orange'
    else:
        return 'red'

# Visualize the hexagons
map_file = visualize_h3_hexagons(df_h3, resolution=9, max_hexagons=500)
```

## Step 6: Querying Restaurants within Specific H3 Cells

One of the key benefits of H3 indexing is the ability to quickly query all locations within a specific cell:

```python
def query_restaurants_in_h3_cell(df_h3, h3_index):
    """Query all restaurants that fall within a specific H3 cell"""
    # Extract the resolution from the H3 index
    resolution = h3.h3_get_resolution(h3_index)
    h3_col = f'h3_index_{resolution}'
    
    # Check if the column exists, if not, add it
    if h3_col not in df_h3.columns:
        df_h3[h3_col] = df_h3.apply(
            lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], resolution), 
            axis=1
        )
    
    # Filter the dataframe
    restaurants_in_cell = df_h3[df_h3[h3_col] == h3_index]
    
    print(f"Found {len(restaurants_in_cell)} restaurants in H3 cell {h3_index}:")
    if not restaurants_in_cell.empty:
        print(restaurants_in_cell[['name', 'country', 'Rcuisine']].head(10))
        if len(restaurants_in_cell) > 10:
            print(f"...and {len(restaurants_in_cell) - 10} more.")
    
    return restaurants_in_cell

# Example: Query restaurants in a specific H3 cell
# Find a populated H3 cell to query
sample_h3 = h3_counts.index[0]  # Get the most populated cell
restaurants_in_cell = query_restaurants_in_h3_cell(df_h3, sample_h3)
```

## Step 7: Performing Spatial Search with H3

H3 provides efficient spatial search capabilities using the k-ring function, which finds all hexagons within a certain distance of a center hexagon:

```python
def perform_spatial_analysis(df_h3, point_lat, point_lng, radius_km=5, resolution=9):
    """Find restaurants within a given radius of a point using H3 k-rings"""
    # Convert the point to an H3 index
    center_h3 = h3.geo_to_h3(point_lat, point_lng, resolution)
    
    # Approximate conversion from km to number of rings
    # Hexagon size varies by resolution
    hex_size_km = 0.174 * 2  # Approximate diameter of res 9 hexagon in km
    k = int(np.ceil(radius_km / hex_size_km))
    
    # Get k-ring indices
    k_ring = h3.k_ring(center_h3, k)
    
    # Filter restaurants within these H3 indices
    h3_col = f'h3_index_{resolution}'
    if h3_col not in df_h3.columns:
        df_h3[h3_col] = df_h3.apply(
            lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], resolution), 
            axis=1
        )
    
    restaurants_in_radius = df_h3[df_h3[h3_col].isin(k_ring)]
    
    print(f"Spatial search around ({point_lat}, {point_lng}) with {radius_km}km radius:")
    print(f"Using H3 resolution {resolution}, k-ring size {k}")
    print(f"Found {len(restaurants_in_radius)} restaurants within radius")
    
    # Create a visualization of the search area
    m = folium.Map(location=[point_lat, point_lng], zoom_start=12)
    
    # Add a marker for the center point
    folium.Marker(
        [point_lat, point_lng],
        popup="Search Center",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    # Add hexagons in the k-ring
    for h3_index in k_ring:
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        folium.Polygon(
            locations=boundary,
            fill=True,
            fill_color='blue',
            fill_opacity=0.2,
            color='black',
            weight=1
        ).add_to(m)
    
    # Add restaurant markers
    for _, row in restaurants_in_radius.iterrows():
        folium.CircleMarker(
            [row['latitude'], row['longitude']],
            radius=3,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.7,
            tooltip=f"{row['name']} - {row['Rcuisine']}"
        ).add_to(m)
    
    # Draw an approximate circle for the search radius
    folium.Circle(
        [point_lat, point_lng],
        radius=radius_km * 1000,  # Convert to meters
        color='red',
        fill=False,
        weight=2
    ).add_to(m)
    
    # Save the map
    output_file = f"spatial_search_map_{radius_km}km_res{resolution}.html"
    m.save(output_file)
    print(f"Search visualization saved to {output_file}")
    
    return restaurants_in_radius, output_file

# Example: Spatial search around a point
# Use a sample location (e.g., Times Square, NYC)
sample_lat, sample_lng = 40.7580, -73.9855  # Times Square coordinates
nearby_restaurants, search_map = perform_spatial_analysis(
    df_h3, sample_lat, sample_lng, radius_km=1, resolution=9
)
```

## Step 8: Converting H3 Data to GeoDataFrame for GIS Analysis

For more advanced GIS analysis, we can convert our H3 data to a GeoDataFrame:

```python
import geopandas as gpd
from shapely.geometry import Polygon

def convert_to_geodataframe(df_h3, resolution=9):
    """Convert H3 indexed data to a GeoDataFrame for GIS operations"""
    h3_col = f'h3_index_{resolution}'
    
    # Ensure we have the H3 index column
    if h3_col not in df_h3.columns:
        df_h3[h3_col] = df_h3.apply(
            lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], resolution), 
            axis=1
        )
    
    # Group by H3 index and count restaurants
    h3_grouped = df_h3.groupby(h3_col).size().reset_index(name='restaurant_count')
    
    # Create a list to store polygon geometries
    polygons = []
    
    # Convert each H3 index to a polygon
    for h3_index in h3_grouped[h3_col]:
        # Get hexagon boundary as a list of [lat, lng] pairs
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        # Convert to a Shapely Polygon
        polygon = Polygon([(lng, lat) for lat, lng in boundary])
        polygons.append(polygon)
    
    # Add the geometry column to create a GeoDataFrame
    h3_grouped['geometry'] = polygons
    gdf = gpd.GeoDataFrame(h3_grouped, geometry='geometry')
    
    # Set the coordinate reference system (CRS) to WGS84
    gdf.crs = "EPSG:4326"
    
    print(f"Created GeoDataFrame with {len(gdf)} H3 hexagons at resolution {resolution}")
    
    # Save the GeoDataFrame to a GeoJSON file
    output_file = f"restaurant_h3_res{resolution}.geojson"
    gdf.to_file(output_file, driver='GeoJSON')
    print(f"GeoJSON saved to {output_file}")
    
    return gdf

# Convert to GeoDataFrame for advanced GIS analysis
gdf = convert_to_geodataframe(df_h3, resolution=9)
```

## Step 9: Analyzing Cuisine Diversity by H3 Cell

Let's perform a more advanced analysis by calculating cuisine diversity for each H3 cell:

```python
from collections import Counter
import numpy as np

def analyze_cuisine_diversity(df_h3, resolution=9):
    """Calculate cuisine diversity for each H3 cell"""
    h3_col = f'h3_index_{resolution}'
    
    # Ensure we have the H3 index column
    if h3_col not in df_h3.columns:
        df_h3[h3_col] = df_h3.apply(
            lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], resolution), 
            axis=1
        )
    
    # Function to calculate diversity (Shannon entropy)
    def shannon_entropy(cuisines):
        counts = Counter(cuisines)
        total = sum(counts.values())
        probabilities = [count/total for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy
    
    # Group by H3 index and calculate diversity
    h3_diversity = df_h3.groupby(h3_col).agg({
        'Rcuisine': list,
        'name': 'count'
    }).reset_index()
    
    # Calculate diversity for each cell
    h3_diversity['cuisine_count'] = h3_diversity['Rcuisine'].apply(
        lambda x: len(set(cuisine for cuisine in x if cuisine))
    )
    h3_diversity['diversity_index'] = h3_diversity['Rcuisine'].apply(
        lambda x: shannon_entropy([cuisine for cuisine in x if cuisine])
    )
    
    # Sort by diversity index
    h3_diversity_sorted = h3_diversity.sort_values('diversity_index', ascending=False)
    
    print("Cuisine diversity analysis:")
    print(f"Average cuisine types per H3 cell: {h3_diversity['cuisine_count'].mean():.2f}")
    print(f"Maximum cuisine diversity index: {h3_diversity['diversity_index'].max():.2f}")
    
    # Show top diverse areas
    print("Top 5 most cuisine-diverse H3 cells:")
    for _, row in h3_diversity_sorted.head(5).iterrows():
        cuisines = Counter([c for c in row['Rcuisine'] if c])
        top_cuisines = ", ".join([f"{cuisine} ({count})" for cuisine, count in cuisines.most_common(3)])
        print(f"H3 index: {row[h3_col]}")
        print(f"  Restaurants: {row['name']}, Cuisine types: {row['cuisine_count']}")
        print(f"  Diversity index: {row['diversity_index']:.2f}")
        print(f"  Top cuisines: {top_cuisines}")
    
    return h3_diversity_sorted

# Analyze cuisine diversity
diversity_data = analyze_cuisine_diversity(df_h3, resolution=9)
```

## Step 10: Integrating H3 with Other Systems

H3 indices can be integrated with other systems and databases for more efficient spatial queries:

### SQL Database Integration

```python
# Example: Storing H3 indices in a SQL database
import sqlite3

# Create a connection to the database
conn = sqlite3.connect('restaurants_h3.db')

# Save the dataframe to a SQL table
df_h3.to_sql('restaurants', conn, if_exists='replace', index=False)

# Create indices on the H3 columns for faster queries
cursor = conn.cursor()
for res in resolutions:
    h3_col = f'h3_index_{res}'
    cursor.execute(f'CREATE INDEX idx_{h3_col} ON restaurants ({h3_col})')
conn.commit()

# Example query: Find all restaurants in a specific H3 cell
h3_index = '89754e64dffffff'  # Example H3 index at resolution 9
query = f"SELECT name, country, Rcuisine FROM restaurants WHERE h3_index_9 = '{h3_index}'"
result = pd.read_sql_query(query, conn)
print(f"Found {len(result)} restaurants in H3 cell {h3_index} via SQL query")
```

### Elasticsearch Integration

```python
# Example: Using H3 with Elasticsearch (pseudocode)
# First, create a mapping with H3 indices as keywords
mapping = {
    "mappings": {
        "properties": {
            "location": {"type": "geo_point"},
            "h3_index_9": {"type": "keyword"},  # Store as keyword for exact matching
            "name": {"type": "text"},
            "cuisine": {"type": "keyword"}
        }
    }
}

# Then index documents with H3 indices
documents = df_h3.apply(
    lambda row: {
        "location": {"lat": row['latitude'], "lon": row['longitude']},
        "h3_index_9": row['h3_index_9'],
        "name": row['name'],
        "cuisine": row['Rcuisine']
    }, 
    axis=1
).tolist()

# Query by H3 index (much faster than geo queries for certain use cases)
query = {
    "term": {
        "h3_index_9": h3_index
    }
}
```

## Step 11: Advanced H3 Features

### Hierarchical Analysis

You can leverage H3's hierarchical structure to perform multi-level analysis:

```python
def hierarchical_analysis(df_h3, location_lat, location_lng):
    """Analyze a location at multiple resolution levels"""
    results = {}
    
    for res in range(5, 13):  # From country level to block level
        h3_index = h3.geo_to_h3(location_lat, location_lng, res)
        h3_col = f'h3_index_{res}'
        
        # Ensure we have the column
        if h3_col not in df_h3.columns:
            df_h3[h3_col] = df_h3.apply(
                lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], res), 
                axis=1
            )
        
        # Count restaurants in this cell
        cell_count = df_h3[df_h3[h3_col] == h3_index].shape[0]
        
        # Get the hexagon size in km²
        hex_area = h3.cell_area(h3_index, unit='km^2')
        
        # Calculate restaurant density
        density = cell_count / hex_area if hex_area > 0 else 0
        
        results[res] = {
            'h3_index': h3_index,
            'resolution': res,
            'restaurant_count': cell_count,
            'hex_area_km2': hex_area,
            'restaurant_density': density
        }
    
    # Print the results
    print("\nHierarchical analysis from region to block level:")
    for res, data in results.items():
        print(f"Resolution {res} ({get_resolution_description(res)}):")
        print(f"  H3 Index: {data['h3_index']}")
        print(f"  Area: {data['hex_area_km2']:.2f} km²")
        print(f"  Restaurants: {data['restaurant_count']}")
        print(f"  Density: {data['restaurant_density']:.2f} restaurants/km²")
    
    return results

def get_resolution_description(resolution):
    """Get a human-readable description of the H3 resolution level"""
    descriptions = {
        0: "Continental",
        1: "Sub-continental",
        2: "Country",
        3: "Large region",
        4: "Region",
        5: "Sub-region",
        6: "City/Metro area",
        7: "District",
        8: "Neighborhood",
        9: "Sub-neighborhood",
        10: "Block",
        11: "Sub-block",
        12: "Building"
    }
    return descriptions.get(resolution, f"Resolution {resolution}")
```

### Polyfill: Converting Arbitrary Shapes to H3 Indices

You can convert arbitrary geographical shapes to H3 indices:

```python
def polyfill_example():
    """Example of converting a polygon to H3 indices"""
    # Define a polygon (e.g., Manhattan bounds)
    polygon = [
        [40.70, -74.02],  # Bottom left
        [40.70, -73.93],  # Bottom right
        [40.83, -73.93],  # Top right
        [40.83, -74.02],  # Top left
        [40.70, -74.02]   # Close the polygon
    ]
    
    # Convert polygon to GeoJSON format expected by h3.polyfill
    geojson_polygon = {
        'type': 'Polygon',
        'coordinates': [[[lng, lat] for lat, lng in polygon]]
    }
    
    # Fill the polygon with H3 indices at resolution 9
    h3_indices = h3.polyfill(geojson_polygon, 9)
    
    print(f"Polygon filled with {len(h3_indices)} H3 hexagons at resolution 9")
    
    # Create a map visualization
    center_lat = sum(p[0] for p in polygon) / len(polygon)
    center_lng = sum(p[1] for p in polygon) / len(polygon)
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)
    
    # Add the original polygon
    folium.Polygon(
        locations=polygon,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.1,
        weight=3
    ).add_to(m)
    
    # Add the H3 hexagons
    for h3_index in h3_indices:
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        folium.Polygon(
            locations=boundary,
            fill=True,
            fill_color='blue',
            fill_opacity=0.3,
            color='blue',
            weight=1
        ).add_to(m)
    
    # Save the map
    m.save("polyfill_example.html")
    
    return h3_indices
```

## Step 12: Performance Optimization

For large datasets, optimization is important:

```python
def optimize_h3_operations(df):
    """Optimize H3 indexing for large datasets"""
    # 1. Use vectorized operations where possible
    # Instead of apply, try vectorized numpy operations for simpler cases
    
    # 2. Process in chunks for very large datasets
    chunk_size = 100000
    total_rows = len(df)
    
    results = []
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        print(f"Processing chunk {start}-{end} of {total_rows}")
        
        chunk = df.iloc[start:end].copy()
        # Process chunk
        chunk['h3_index_9'] = chunk.apply(
            lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], 9), 
            axis=1
        )
        results.append(chunk)
    
    # Combine results
    df_h3 = pd.concat(results)
    
    # 3. Use parallel processing for CPU-intensive operations
    # (Requires additional libraries like multiprocessing or joblib)
    
    return df_h3
```

## Conclusion

You've now learned how to:
1. Implement H3 indexing for geospatial data
2. Analyze the distribution of data across H3 cells
3. Visualize H3 hexagons on interactive maps
4. Perform spatial queries using H3
5. Convert H3 data to GeoDataFrame for GIS analysis
6. Calculate metrics like cuisine diversity by H3 cell
7. Use advanced H3 features like hierarchical analysis and polyfill

This implementation provides a solid foundation for geospatial analysis of your restaurant data. The H3 indexing system offers significant performance advantages for spatial queries and aggregations compared to traditional methods.

## Next Steps

1. Integrate this H3 implementation with your existing data pipeline
2. Create interactive dashboards using the H3 visualizations
3. Develop API endpoints for H3-based spatial queries
4. Explore machine learning applications using H3 features (e.g., predicting restaurant density or cuisine diversity for new areas)
5. Consider implementing real-time updates to your H3 indices as new data becomes available