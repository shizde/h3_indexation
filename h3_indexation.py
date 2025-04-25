"""
H3 Indexing Implementation for Restaurant Geospatial Data
=======================================================

This script demonstrates how to implement H3 indexing for geospatial data,
specifically for a restaurant dataset with latitude and longitude coordinates.

Compatible with H3 version 4.x

Steps:
1. Install required packages
2. Load and prepare the data
3. Implement H3 indexing at different resolution levels
4. Analyze data distribution across H3 cells
5. Visualize the H3 hexagons on a map
6. Query restaurants within specific H3 cells
7. Perform spatial analysis using H3 functionality

Requirements:
- Python 3.8+
- pandas
- h3 (version 4.x)
- folium
- matplotlib
- geopandas
"""

# Step 1: Install required packages
# ---------------------------------
# Run these commands in your terminal:
# pip install pandas h3 folium matplotlib geopandas shapely

import pandas as pd
import h3
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import json
from collections import Counter

# Print the h3 version to verify
print(f"Using h3 version: {h3.__version__}")

# Step 2: Load and prepare the data
# ---------------------------------
def load_data(file_path):
    """Load the CSV data and handle any preprocessing needs"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Check for missing values in latitude/longitude
    missing_coords = df[df['latitude'].isna() | df['longitude'].isna()]
    if not missing_coords.empty:
        print(f"Warning: Found {len(missing_coords)} records with missing coordinates.")
        # Drop rows with missing coordinates if needed
        df = df.dropna(subset=['latitude', 'longitude'])

    print(f"Loaded {len(df)} records with valid coordinates.")

    # Basic data statistics
    print("\nData Overview:")
    print(f"Latitude range: {df['latitude'].min()} to {df['latitude'].max()}")
    print(f"Longitude range: {df['longitude'].min()} to {df['longitude'].max()}")

    # Display country distribution
    country_counts = df['country'].value_counts().head(10)
    print("\nTop 10 countries by restaurant count:")
    print(country_counts)

    # Display cuisine distribution
    cuisine_counts = df['Rcuisine'].value_counts().head(10)
    print("\nTop 10 cuisines:")
    print(cuisine_counts)

    return df

# Step 3: Implement H3 indexing at different resolution levels
# -----------------------------------------------------------
def add_h3_indices(df, resolutions=[7, 8, 9, 10]):
    """Add H3 indices at specified resolution levels to the dataframe"""
    print(f"\nGenerating H3 indices at resolutions: {resolutions}...")

    # Create a copy of the dataframe to avoid modifying the original
    df_h3 = df.copy()

    # Add H3 index columns for each resolution
    for res in resolutions:
        col_name = f'h3_index_{res}'

        # Use latlng_to_cell instead of geo_to_h3
        df_h3[col_name] = df_h3.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], res),
            axis=1
        )

        # Count unique hexagons at this resolution
        unique_hexagons = df_h3[col_name].nunique()
        print(f"Resolution {res}: {unique_hexagons} unique hexagons")

    return df_h3

# Step 4: Analyze data distribution across H3 cells
# ------------------------------------------------
def analyze_h3_distribution(df_h3, resolution=9):
    """Analyze the distribution of restaurants across H3 cells"""
    h3_col = f'h3_index_{resolution}'

    # Count restaurants per H3 cell
    h3_counts = df_h3[h3_col].value_counts()

    print(f"\nAnalyzing restaurant distribution at resolution {resolution}:")
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

    # Return the full distribution data for potential visualization
    return h3_counts

# Step 5: Visualize the H3 hexagons on a map
# ------------------------------------------
def visualize_h3_hexagons(df_h3, resolution=9, max_hexagons=1000):
    """Create a folium map visualizing H3 hexagons with restaurant density"""
    h3_col = f'h3_index_{resolution}'

    # Get the most populated hexagons for visualization
    h3_counts = df_h3[h3_col].value_counts()

    # Limit to max_hexagons to avoid browser performance issues
    if len(h3_counts) > max_hexagons:
        print(f"\nLimiting visualization to the {max_hexagons} most populated hexagons.")
        h3_counts = h3_counts.head(max_hexagons)

    # Calculate the center point for the map based on data median
    center_lat = df_h3['latitude'].median()
    center_lng = df_h3['longitude'].median()

    # Create a map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=4)

    # Add hexagons to the map
    for h3_index, count in h3_counts.items():
        # Get hexagon boundary as a list of [lat, lng] pairs
        boundary = h3.cell_to_boundary(h3_index)

        # Convert to the format expected by folium (list of [lat, lng] pairs)
        boundary_folium = [[lat, lng] for lat, lng in boundary]

        # Calculate color based on restaurant count (using a simple scale)
        color = get_color_for_count(count, h3_counts.max())

        # Create a polygon for each hexagon
        folium.Polygon(
            locations=boundary_folium,
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
    print(f"\nMap saved to {output_file}")

    return output_file

def get_color_for_count(count, max_count):
    """Generate a color on a red-yellow-green scale based on count"""
    # Scale from 0 to 1
    ratio = count / max_count

    # Create a color scale (green for low, yellow for medium, red for high)
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

# Step 6: Query restaurants within specific H3 cells
# -------------------------------------------------
def query_restaurants_in_h3_cell(df_h3, h3_index):
    """Query all restaurants that fall within a specific H3 cell"""
    # Extract the resolution from the H3 index
    resolution = h3.get_resolution(h3_index)
    h3_col = f'h3_index_{resolution}'

    # Check if the column exists, if not, add it
    if h3_col not in df_h3.columns:
        df_h3[h3_col] = df_h3.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], resolution),
            axis=1
        )

    # Filter the dataframe
    restaurants_in_cell = df_h3[df_h3[h3_col] == h3_index]

    print(f"\nFound {len(restaurants_in_cell)} restaurants in H3 cell {h3_index}:")
    if not restaurants_in_cell.empty:
        print(restaurants_in_cell[['name', 'country', 'Rcuisine']].head(10))
        if len(restaurants_in_cell) > 10:
            print(f"...and {len(restaurants_in_cell) - 10} more.")

    return restaurants_in_cell

# Step 7: Perform spatial analysis using H3 functionality
# ------------------------------------------------------
def perform_spatial_analysis(df_h3, point_lat, point_lng, radius_km=5, resolution=9):
    """Find restaurants within a given radius of a point using H3 k-rings"""
    # Convert the point to an H3 index
    center_h3 = h3.latlng_to_cell(point_lat, point_lng, resolution)

    # Get H3 indexes within the radius (k-ring)
    # Calculate k (number of rings) based on radius and resolution
    # Hexagon size varies by resolution, so we need to approximate

    # Approximate conversion from km to number of rings
    # This is a rough estimate and may need adjustment based on actual geography
    hex_size_km = 0.174 * 2  # Approximate diameter of res 9 hexagon in km
    k = int(np.ceil(radius_km / hex_size_km))

    # Get k-ring indices using grid_disk instead of k_ring
    k_ring = h3.grid_disk(center_h3, k)

    # Filter restaurants within these H3 indices
    h3_col = f'h3_index_{resolution}'
    if h3_col not in df_h3.columns:
        df_h3[h3_col] = df_h3.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], resolution),
            axis=1
        )

    restaurants_in_radius = df_h3[df_h3[h3_col].isin(k_ring)]

    print(f"\nSpatial search around ({point_lat}, {point_lng}) with {radius_km}km radius:")
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
        boundary = h3.cell_to_boundary(h3_index)
        # Convert to the format expected by folium
        boundary_folium = [[lat, lng] for lat, lng in boundary]

        folium.Polygon(
            locations=boundary_folium,
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

# Step 8: Convert H3 data to GeoDataFrame for advanced GIS analysis
# ----------------------------------------------------------------
def convert_to_geodataframe(df_h3, resolution=9):
    """Convert H3 indexed data to a GeoDataFrame for GIS operations"""
    h3_col = f'h3_index_{resolution}'

    # Ensure we have the H3 index column
    if h3_col not in df_h3.columns:
        df_h3[h3_col] = df_h3.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], resolution),
            axis=1
        )

    # Group by H3 index and count restaurants
    h3_grouped = df_h3.groupby(h3_col).size().reset_index(name='restaurant_count')

    # Create a list to store polygon geometries
    polygons = []

    # Convert each H3 index to a polygon
    for h3_index in h3_grouped[h3_col]:
        # Get hexagon boundary and convert to the format needed for Shapely
        boundary = h3.cell_to_boundary(h3_index)
        # Shapely needs coordinates as (lng, lat) not (lat, lng)
        polygon = Polygon([(lng, lat) for lat, lng in boundary])
        polygons.append(polygon)

    # Add the geometry column to create a GeoDataFrame
    h3_grouped['geometry'] = polygons
    gdf = gpd.GeoDataFrame(h3_grouped, geometry='geometry')

    # Set the coordinate reference system (CRS) to WGS84
    gdf.crs = "EPSG:4326"

    print(f"\nCreated GeoDataFrame with {len(gdf)} H3 hexagons at resolution {resolution}")

    # Save the GeoDataFrame to a GeoJSON file
    output_file = f"restaurant_h3_res{resolution}.geojson"
    gdf.to_file(output_file, driver='GeoJSON')
    print(f"GeoJSON saved to {output_file}")

    return gdf

# Step 9: Calculate cuisine diversity by H3 cell
# ---------------------------------------------
def analyze_cuisine_diversity(df_h3, resolution=9):
    """Calculate cuisine diversity for each H3 cell"""
    h3_col = f'h3_index_{resolution}'

    # Ensure we have the H3 index column
    if h3_col not in df_h3.columns:
        df_h3[h3_col] = df_h3.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], resolution),
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

    print("\nCuisine diversity analysis:")
    print(f"Average number of cuisine types per H3 cell: {h3_diversity['cuisine_count'].mean():.2f}")
    print(f"Maximum cuisine diversity index: {h3_diversity['diversity_index'].max():.2f}")

    # Show top diverse areas
    print("\nTop 5 most cuisine-diverse H3 cells:")
    for _, row in h3_diversity_sorted.head(5).iterrows():
        cuisines = Counter([c for c in row['Rcuisine'] if c])
        top_cuisines = ", ".join([f"{cuisine} ({count})" for cuisine, count in cuisines.most_common(3)])
        print(f"H3 index: {row[h3_col]}")
        print(f"  Restaurants: {row['name']}, Cuisine types: {row['cuisine_count']}")
        print(f"  Diversity index: {row['diversity_index']:.2f}")
        print(f"  Top cuisines: {top_cuisines}")

    return h3_diversity_sorted

# Step 10: Advanced H3 features - Polyfill for arbitrary shapes
# ------------------------------------------------------------
def polyfill_example(boundary_coords, resolution=9):
    """Convert an arbitrary polygon to H3 indices"""
    # Convert boundary to GeoJSON format expected by h3.polyfill
    geojson_polygon = {
        'type': 'Polygon',
        'coordinates': [[[lng, lat] for lat, lng in boundary_coords]]
    }

    # Fill the polygon with H3 indices using polyfill
    h3_indices = h3.polyfill(geojson_polygon, resolution)

    print(f"\nPolygon filled with {len(h3_indices)} H3 hexagons at resolution {resolution}")

    return h3_indices

# Step 11: Hierarchical Analysis
# -----------------------------
def hierarchical_analysis(df_h3, location_lat, location_lng):
    """Analyze a location at multiple resolution levels"""
    results = {}

    for res in range(5, 13):  # From country level to block level
        h3_index = h3.latlng_to_cell(location_lat, location_lng, res)
        h3_col = f'h3_index_{res}'

        # Ensure we have the column
        if h3_col not in df_h3.columns:
            df_h3[h3_col] = df_h3.apply(
                lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], res),
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

# Main function to execute the workflow
# ------------------------------------
def main():
    """Main function to run the complete H3 indexing workflow"""
    # Load the restaurant data
    file_path = 'synthesized_restaurants.csv'
    df = load_data(file_path)

    # Add H3 indices at different resolutions
    resolutions = [6, 9, 12]  # Low, medium, and high resolutions
    df_h3 = add_h3_indices(df, resolutions)

    # Save the indexed data for future use
    df_h3.to_csv('restaurants_h3_indexed.csv', index=False)

    # Analyze distribution at resolution 9 (medium granularity)
    h3_counts = analyze_h3_distribution(df_h3, resolution=9)

    # Visualize the hexagons
    map_file = visualize_h3_hexagons(df_h3, resolution=9, max_hexagons=500)

    # Example: Query restaurants in a specific H3 cell
    # Find a populated H3 cell to query
    sample_h3 = h3_counts.index[0]  # Get the most populated cell
    restaurants_in_cell = query_restaurants_in_h3_cell(df_h3, sample_h3)

    # Example: Spatial search
    # Use a popular location (e.g., Times Square, NYC)
    sample_lat, sample_lng = 40.7580, -73.9855  # Times Square coordinates
    nearby_restaurants, search_map = perform_spatial_analysis(
        df_h3, sample_lat, sample_lng, radius_km=1, resolution=9
    )

    # Convert to GeoDataFrame for advanced GIS analysis
    gdf = convert_to_geodataframe(df_h3, resolution=9)

    # Analyze cuisine diversity
    diversity_data = analyze_cuisine_diversity(df_h3, resolution=9)

    # Example: Hierarchical analysis
    # Using a popular location (e.g., Central Park, NYC)
    central_park_lat, central_park_lng = 40.7812, -73.9665
    hierarchical_results = hierarchical_analysis(df_h3, central_park_lat, central_park_lng)

    print("\nH3 indexing workflow completed!")
    print("Files generated:")
    print(f"- restaurants_h3_indexed.csv (Data with H3 indices)")
    print(f"- {map_file} (H3 visualization)")
    print(f"- {search_map} (Spatial search visualization)")
    print("- restaurant_h3_res9.geojson (GeoJSON with H3 data)")

if __name__ == "__main__":
    main()