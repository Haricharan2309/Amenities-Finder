# Updated app.py to combine similar categories, improve precision, and add amenity ratings
from flask import Flask, render_template, request
import googlemaps
import json
import openai
import math

app = Flask(__name__)

GOOGLE_MAPS_API_KEY = "API_Key"  # Replace with your actual key
OPENAI_API_KEY = "API_KEY_2"  # Replace with your actual key

# Google Maps and OpenAI client setup
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
openai.api_key = OPENAI_API_KEY

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        address = request.form["address"]
        amenities_by_type, report = generate_report(address)
        return render_template("index.html", address=address, amenities=amenities_by_type, report=report)
    return render_template("index.html")

def generate_report(address):
    # 1. Geocode the address
    geocode_result = gmaps.geocode(address)
    if not geocode_result:
        return {}, "Address not found."
    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lng = geocode_result[0]["geometry"]["location"]["lng"]

    # 2. Define amenity types with more relevant categories
    amenity_types = {
        "school": "Schools and Universities",
        "university": "Schools and Universities",
        "park": "Parks and Green Spaces",
        "hospital": "Hospitals and Medical Centers",
        "grocery_or_supermarket": "Grocery Stores and Markets",
        "transit_station": "Public Transport Hubs",
        "bus_station": "Public Transport Hubs",
        "train_station": "Public Transport Hubs",
        "gym": "Gyms and Fitness Centers",
        "restaurant": "Restaurants and Cafés",
        "cafe": "Restaurants and Cafés",
        "shopping_mall": "Shopping Centers and Malls"
    }

    # 3. Fetch 3 amenities for each type (within 10 miles)
    amenities_by_type = {}
    for amenity_type, category_name in amenity_types.items():
        places_result = gmaps.places_nearby(
            location=(lat, lng),
            radius=16093.4,  # 10 miles in meters
            type=amenity_type,
            keyword="fast food" if amenity_type == "fast_food" else ""
        )
        # Calculate distance in miles and filter by rating for each amenity
        filtered_amenities = []
        for amenity in places_result.get("results", []):
            if amenity.get("business_status") != "OPERATIONAL":
                continue
            amenity['distance'] = calculate_distance(lat, lng, amenity['geometry']['location']['lat'], amenity['geometry']['location']['lng']) * 0.621371  # Convert to miles
            if "rating" in amenity and amenity["rating"] >= 4.0 and amenity.get("user_ratings_total", 0) >= 100:
                # Validate amenity category using LLM
                validation_prompt = f"""
                Validate whether the following place name belongs to the category '{category_name}':
                Name: {amenity['name']}
                Type of Category: {category_name}

                Respond with 'Yes' or 'No' only.
                """
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an assistant validating the correctness of amenity categories."},
                        {"role": "user", "content": validation_prompt}
                    ],
                    max_tokens=5,
                    temperature=0.0,
                )
                validation_result = response['choices'][0]['message']['content'].strip().lower()
                if validation_result == 'yes':
                    filtered_amenities.append(amenity)

        # Sort by distance and take top 3
        sorted_amenities = sorted(filtered_amenities, key=lambda a: a['distance'])[:3]
        if sorted_amenities:
            amenities_by_type[category_name] = sorted_amenities

    # 4. Construct the report prompt with a formatted list for each amenity type
    amenity_list = ""
    for category_name, amenities in amenities_by_type.items():
        if amenities:
            amenity_list += f"\n{category_name}:\n"
            for a in amenities:
                amenity_list += f"{a['name']} - {a['distance']:.2f} miles, Rating: {a.get('rating', 'N/A')}, "
            amenity_list = amenity_list[:-2]  # Remove trailing comma and space

    prompt_data = f"""
    Analyze the following local amenities near this address: {address}

    {amenity_list}

    Write ONLY a concise summary (3-4 lines) of whether it's a good place to live based on these amenities.
    """

    # 5. Invoke OpenAI API to generate a summary
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_data}
        ],
        max_tokens=150,
        temperature=0.7,
        top_p=0.9
    )
    report = response['choices'][0]['message']['content'].strip()

    return amenities_by_type, report

# Helper function to calculate distance (using Haversine formula)
def calculate_distance(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

if __name__ == "__main__":
    app.run(debug=True)
