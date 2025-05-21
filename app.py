import streamlit as st
import numpy as np
import pandas as pd
import base64
import pickle
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def main():
    
    # Set page config
    st.set_page_config(page_title="TourEase", layout="wide")

    # CSS for horizontal menu
    st.markdown(
        """
        <style>
        .menu-container {
            display: flex;
            justify-content: center;
            margin-top: 10px;
            margin-bottom: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        input[type="radio"] {
            display: none;
        }
        label {
            background: #000;
            padding: 10px 30px;
            margin-right: 5px;
            border-radius: 10px 10px 0 0;
            cursor: pointer;
            font-weight: bold;
            color: #444;
            transition: background-color 0.3s ease, color 0.3s ease;
            user-select: none;
        }
        input[type="radio"]:checked + label {
            background: #4a90e2;
            color: white;
            box-shadow: 0 4px 10px rgb(74 144 226 / 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    menu = st.radio(
        "",
        ("Home", "Explore", "Rating Predictions", "VisitMode Prediction","Recommendation System"),
        key="menu",
        horizontal=True,
    )

    def set_bg_image(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    if menu == "Home":
        set_bg_image(r"C:\Tourism\assetsbanner.png")
        st.markdown(
            """
            <div style='display: flex; justify-content: space-between; align-items: center; padding: 10px 30px; background-color: rgba(0,0,0,0.4); border-bottom: 1px solid #fff;'>
                <div style='font-size: 32px; color: #FFD700; font-weight: bold;'>TourEase!</div>
                <div style='color: #fff; font-size: 18px;'></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style='text-align: center; margin-top: 180px; color: white;'>
                <h1 style='font-size: 60px; font-weight: bold;'>Enjoy your dream vacation</h1>
                <h3 style='font-size: 32px;'>Explore new horizons and create lasting memories.</h3>
            </div>
            <div style='width: 50%; margin-left: auto; margin-right: 60px; margin-top: 40px;
                        font-size: 20px; color: black; line-height: 1.8; text-align: left;
                        background-color: rgba(255,255,255,0.75); padding: 20px; border-radius: 12px;'>
                <p>📍 <b>Data-powered experiences</b> | <b>Personalized Travel</b> | <b>AI-enabled Support</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif menu == "Explore":
        st.markdown(
            """
            <style>
            body, .stApp {
                background-color: #e6f2ff;
                color: black !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .stSidebar {
                background-color: #cce6ff;
                padding: 20px;
                color: black !important;
            }
            .attraction-card {
                background: white;
                border-radius: 12px;
                padding: 15px;
                box-shadow: 2px 2px 8px #a3c2ff;
                margin-bottom: 25px;
                transition: transform 0.2s ease;
                color: black !important;
            }
            .attraction-card:hover {
                transform: scale(1.03);
                box-shadow: 4px 4px 12px #7bb1ff;
            }
            .attraction-image {
                border-radius: 10px;
                width: 100%;
                max-height: 180px;
                object-fit: cover;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        data = {
            "Attraction_Name": ["Sunny Beach", "Mountain Peak", "Historic Museum", "City Park", "Adventure Land", "Serene Lake"],
            "Category": ["Beach", "Mountain", "Museum", "Park", "Adventure", "Lake"],
            "Location": ["California", "Colorado", "New York", "California", "Nevada", "Minnesota"],
            "Rating": [4.5, 4.7, 4.3, 4.0, 4.8, 4.6],
            "Description": [
                "A beautiful sunny beach perfect for family outings.",
                "A challenging mountain with breathtaking views.",
                "Museum showcasing local history and art.",
                "Lush green city park with playgrounds and trails.",
                "Theme park with thrilling rides and fun activities.",
                "Peaceful lake ideal for fishing and relaxation."
            ],
            "Image_URL": [
                "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=400&q=60",
                "https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=400&q=60",
                "https://images.unsplash.com/photo-1526483360697-3e91f9df3f6e?auto=format&fit=crop&w=400&q=60",
                "https://images.unsplash.com/photo-1500534623283-312aade485b7?auto=format&fit=crop&w=400&q=60",
                "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?auto=format&fit=crop&w=400&q=60",
                "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?auto=format&fit=crop&w=400&q=60"
            ]
        }
        df = pd.DataFrame(data)

        st.sidebar.header("Filter Attractions")
        selected_categories = st.sidebar.multiselect("Select Category", df["Category"].unique(), default=df["Category"].unique())
        selected_locations = st.sidebar.multiselect("Select Location", df["Location"].unique(), default=df["Location"].unique())
        min_rating = st.sidebar.slider("Minimum Rating", min_value=0.0, max_value=5.0, value=3.5, step=0.1)

        filtered_df = df[
            (df["Category"].isin(selected_categories)) &
            (df["Location"].isin(selected_locations)) &
            (df["Rating"] >= min_rating)
        ]

        st.title("🌍 Explore Top Tourist Attractions")

        if filtered_df.empty:
            st.warning("No attractions found matching your criteria. Please adjust your filters.")
        else:
            cols = st.columns(2)
            for idx, row in filtered_df.iterrows():
                col = cols[idx % 2]
                with col:
                    st.markdown(
                        f"""
                        <div class="attraction-card">
                            <img src="{row['Image_URL']}" alt="{row['Attraction_Name']}" class="attraction-image" />
                            <h3>{row['Attraction_Name']}</h3>
                            <p><b>Category:</b> {row['Category']}</p>
                            <p><b>Location:</b> {row['Location']}</p>
                            <p><b>Rating:</b> {row['Rating']} ⭐</p>
                            <p>{row['Description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    elif menu == "Rating Predictions":
       st.markdown(
        """
       <style>
       /* Make the main content take full viewport height */
       .full-height {
           height: 100vh;
           overflow: hidden;
           display: flex;
           flex-direction: row;
       }
           /* Left column with image full height */
        .left-col > div {
           height: 100vh !important;
           padding: 0 !important;
       }
        /* Right column padding and vertical scroll if needed */
            .right-col > div {
            height: 100vh !important;
            overflow-y: auto;
            padding-left: 30px;
            padding-top: 30px;
            padding-right: 30px;
        }
        /* Make image fill left column */
        .left-col img {
           object-fit: cover;
           width: 100% !important;
           height: 100vh !important;
        }

        /* Larger fonts and inputs */
        .title {
           font-size: 42px !important;
           font-weight: bold;
       }
       .subheader {
           font-size: 28px !important;
           font-weight: 600;
           margin-bottom: 10px;
       }
        .sentence-text {
           font-size: 22px;
           margin-bottom: 8px;
       }
        div.stSelectbox > div > div > select, 
        div.stNumberInput > div > input {
           font-size: 22px;
           height: 45px;
       }
        button.css-18ni7ap.e8zbici2 {
           font-size: 24px;
           padding: 10px 24px;
           margin-top: 20px;
       }
       </style>
       """,
       unsafe_allow_html=True,
       )

       @st.cache_resource
       def load_resources():
           with open(r"C:\Tourism\combined_encoder.pkl", "rb") as f:
               encoders = pickle.load(f)
           with open(r"C:\Tourism\final_dummy_columns.pkl", "rb") as f:
               final_features = pickle.load(f)  # Load your final dummy columns here
           with open(r"C:\Tourism\attraction_rating_model.pkl", "rb") as f:
               model = pickle.load(f)
           with open(r"C:\Tourism\scaler.pkl", "rb") as f:
               scaler = pickle.load(f)
           df = pd.read_csv(r"C:\Tourism\Added_data.csv")
           return encoders, final_features, model, scaler, df

       encoders, final_features, model, scaler, df = load_resources()

# Create two columns for full height layout
       col1, col2 = st.columns([1, 3], gap="small")
       col1, col2 = st.columns([1,3], gap="small")

       with col1:
    # Add CSS class for left column
           st.markdown('<div class="left-col">', unsafe_allow_html=True)
           image = Image.open(r"C:\Tourism\star_rating__w820__h462__.jpg")
           st.image(image, use_container_width=True)
           st.markdown('</div>', unsafe_allow_html=True)



       with col2:
           st.markdown('<div class="right-col">', unsafe_allow_html=True)
           st.markdown('<div class="title">⭐ Smart Attraction Rating Predictions</div>', unsafe_allow_html=True)
           st.markdown('<div class="subheader">🧾 Please provide the following information:</div>', unsafe_allow_html=True)

           st.markdown('<div class="sentence-text">🌍 Choose the continent where the attraction is located:</div>', unsafe_allow_html=True)
           continent_list = sorted(df["Continent"].unique().tolist())
           selected_continent = st.selectbox("", continent_list, key="continent")

           filtered_df_continent = df[df["Continent"] == selected_continent]

           st.markdown('<div class="sentence-text">🏙️ Choose the city for the attraction:</div>', unsafe_allow_html=True)
           city_list = sorted(filtered_df_continent["CityName"].unique().tolist())
           selected_city = st.selectbox("", city_list, key="city")

           filtered_df_city = filtered_df_continent[filtered_df_continent["CityName"] == selected_city]

           st.markdown('<div class="sentence-text">📍 Select the specific attraction you want to rate:</div>', unsafe_allow_html=True)
           attraction_list = sorted(filtered_df_city["Attraction"].unique().tolist())
           selected_attraction = st.selectbox("", attraction_list, key="attraction")

           filtered_df_attr = filtered_df_city[filtered_df_city["Attraction"] == selected_attraction]

           st.markdown('<div class="sentence-text">🌦️ Select the season of the country:</div>', unsafe_allow_html=True)
           season_list = sorted(filtered_df_attr["CountrySeason"].unique().tolist())
           selected_country_season = st.selectbox("", season_list, key="season")

           if st.button("Predict Rating"):
               try:
                  sample_row = filtered_df_attr[filtered_df_attr["CountrySeason"] == selected_country_season].iloc[0]

                  user_inputs = {
                       "UserAvgRating": sample_row["UserAvgRating"],
                       "AttractionAvgRating": sample_row["AttractionAvgRating"],
                       "Attraction": selected_attraction,
                       "UserVisitCount": sample_row["UserVisitCount"],
                       "VisitYear": sample_row["VisitYear"],
                       "AttractionPopularity": sample_row["AttractionPopularity"],
                       "CountrySeason": selected_country_season,
                       "CityName": selected_city,
                       "Continent": selected_continent
                   }

                  country_season_le = encoders["label_countryseason"].transform([user_inputs["CountrySeason"]])[0]
                  attraction_le = encoders["label_attraction"].transform([user_inputs["Attraction"]])[0]
                  city_le = encoders["label_city"].transform([user_inputs["CityName"]])[0]

                  base_features = {
                       "UserAvgRating": user_inputs["UserAvgRating"],
                       "AttractionAvgRating": user_inputs["AttractionAvgRating"],            
                       "CountrySeason": country_season_le,
                       "Attraction": attraction_le,
                       "UserVisitCount": user_inputs["UserVisitCount"],
                       "VisitYear": user_inputs["VisitYear"],
                       "AttractionPopularity": user_inputs["AttractionPopularity"],
                       "CityName": city_le
                   }

                  df_base = pd.DataFrame([base_features])

                  continent_encoded = encoders["onehot"].transform([[user_inputs["Continent"]]])
                  continent_cols = encoders["onehot"].get_feature_names_out(["Continent"])
                  df_continent = pd.DataFrame(continent_encoded, columns=continent_cols)

                  full_df = pd.concat([df_base, df_continent], axis=1)

            # Use final_dummy_columns.pkl here for ordering columns
                  full_df = full_df.reindex(columns=final_features, fill_value=0)

                  num_cols = ["UserAvgRating", "AttractionAvgRating", "UserVisitCount", "VisitYear", "AttractionPopularity"]
                  full_df[num_cols] = scaler.transform(full_df[num_cols])

                  prediction = model.predict(full_df)

                  st.success(f"🌟 Predicted Rating: {prediction[0]:.1f}")

               except Exception as e:
                   st.error(f"⚠️ Error during prediction: {e}")

       st.markdown('</div>', unsafe_allow_html=True)

    elif menu == 'VisitMode Prediction':      

       st.title("🧭 Visit Mode Prediction")

# ✅ Load all resources with caching
       @st.cache_resource
       def load_visitmode_resources():
            with open("visit_features_scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open("visitmode_label_encoder.pkl", "rb") as f:
                le_visitmode = pickle.load(f)
            with open("random_forest_visitmode_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("visit_dummy_columns.pkl", "rb") as f:
                expected_columns = pickle.load(f)

            df_original = pd.read_csv("Visitselected_features.csv")
            df_scaled = pd.read_csv("encoded_scaled_visit_data.csv")

            return scaler, le_visitmode, model, expected_columns, df_original, df_scaled

# ✅ Load data and models
       try:
            scaler, le_visitmode, model, expected_columns, df_original, df_scaled = load_visitmode_resources()
       except Exception as e:
            st.error(f"Error loading resources: {e}")
            st.stop()

# ✅ Streamlit UI inputs
       user_ids = df_original['UserId'].unique()
       selected_user = st.selectbox("Select User ID", user_ids)

       attractions_for_user = df_original[df_original['UserId'] == selected_user]['Attraction'].unique()
       selected_attraction = st.selectbox("Select Attraction", attractions_for_user)

       visit_months = sorted(df_original[
            (df_original['UserId'] == selected_user) &
            (df_original['Attraction'] == selected_attraction)
       ]['VisitMonth'].unique())
       selected_month = st.selectbox("Select Visit Month", visit_months)
       
       row_index = []
# ✅ Predict button
       if st.button("Predict Visit Mode"):
            row_index = df_original[
                (df_original['UserId'] == selected_user) &
                (df_original['Attraction'] == selected_attraction) &
                (df_original['VisitMonth'] == selected_month)
            ].index

            if len(row_index) == 0:
                st.error("No matching data found.")
            else:
                idx = row_index[0]
                X = df_scaled.loc[idx].copy()

        # Drop unwanted columns if they exist
                columns_to_drop = [col for col in ['UserId', 'Attraction', 'VisitMode'] if col in X.index]
                X_input = X.drop(columns_to_drop)

        # Ensure it's a 2D DataFrame
                X_input = pd.DataFrame([X_input])

        # Match expected columns
                X_input = X_input.reindex(columns=expected_columns, fill_value=0)

        # Predict
                pred_encoded = model.predict(X_input)[0]
                pred_label = le_visitmode.inverse_transform([pred_encoded])[0]

        # Results
                st.markdown("### ✅ Predicted Visit Mode:")
                st.success(pred_label)

                st.markdown("### 🔎 Feature Values Used:")
                st.write(df_original.loc[idx])

    elif menu == "Recommendation System":
        df = pd.read_csv("Tourism_preprocessed_data.csv")

# Prepare numeric features
        numeric_cols = [
          "AttractionPopularity", "IsTopAttraction", "RegionPopularity",
          "IsCulturalAttraction", "CityAvgRating", "CountryAvgRating",
          "AttractionRatingStd", "AttractionAvgRating", "AttractionRatingDeviation",
          "AttractionSeasonality", "PopularityRatio"
        ]

# Convert to numeric and fill missing before grouping
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Group by attraction
        attraction_features = df.groupby("Attraction")[numeric_cols].mean().reset_index()
        attraction_features[numeric_cols] = attraction_features[numeric_cols].fillna(0)

# Normalize
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(attraction_features[numeric_cols])
        assert not np.isnan(normalized).any(), "NaNs still exist in normalized matrix!"

# Cosine similarity
        similarity_matrix = cosine_similarity(normalized)
        attraction_names = attraction_features["Attraction"].tolist()

# Content-Based Recommendations
        def content_based_recommendations(attraction_name, top_n=5):
            if attraction_name not in attraction_names:
               return []
            idx = attraction_names.index(attraction_name)
            similarity_scores = list(enumerate(similarity_matrix[idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            top_indices = [i for i, score in similarity_scores[1:top_n+1]]
            return [attraction_names[i] for i in top_indices]

# Collaborative Filtering
        user_item_matrix = df.pivot_table(index="UserId", columns="Attraction", values="Rating").fillna(0)
        item_similarity = cosine_similarity(user_item_matrix.T)
        item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

        def collaborative_recommendations(user_id, top_n=5):
            if user_id not in user_item_matrix.index:
               return []
            user_ratings = user_item_matrix.loc[user_id]
            scores = item_similarity_df.dot(user_ratings).div(item_similarity_df.sum(axis=1))
            scores = scores.sort_values(ascending=False)
            already_visited = user_ratings[user_ratings > 0].index.tolist()
            recommended = [attr for attr in scores.index if attr not in already_visited]
            return recommended[:top_n]

# Hybrid Recommendations
        def hybrid_recommendations(user_id, attraction_name, top_n=5):
            content_rec = content_based_recommendations(attraction_name, top_n=10)
            collab_rec = collaborative_recommendations(user_id, top_n=10)
            combined = list(set(content_rec) & set(collab_rec))
            if len(combined) < top_n:
               combined += [item for item in content_rec + collab_rec if item not in combined]
            return combined[:top_n]

# ============ Streamlit UI =============
        st.title("🏖️ Tourism Recommendation System")

# Sidebar
        st.sidebar.header("User Input")
        user_id = st.sidebar.selectbox("Select User ID", sorted(df["UserId"].unique()))
        attraction = st.sidebar.selectbox("Choose Attraction", sorted(df["Attraction"].unique()))
        rec_type = st.sidebar.radio("Recommendation Type", ["Content-Based", "Collaborative", "Hybrid"])

# Generate Recommendations
        if st.sidebar.button("Get Recommendations"):
           
            if rec_type == "Content-Based":
                recs = content_based_recommendations(attraction)
        elif rec_type == "Collaborative":
            recs = collaborative_recommendations(user_id)
        else:
            recs = hybrid_recommendations(user_id, attraction)

        st.subheader(f"🔍 Top Recommendations ({rec_type})")
        if recs:
            for i, rec in enumerate(recs, 1):
                st.write(f"{i}. {rec}")
        else:
            st.warning("No recommendations found.")


if __name__ == "__main__":
    main()