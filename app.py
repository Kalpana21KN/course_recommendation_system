import pickle
import pandas as pd
import streamlit as st

# Load saved objects
df = pd.read_csv("courses_processed.csv")

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

# Recommendation function
def hybrid_recommend(course_name, top_n=5, w1=0.7, w2=0.3):
    try:
        idx = df[df['course_name'].str.lower() == course_name.lower()].index[0]
    except IndexError:
        return pd.DataFrame({"Error": ["Course not found"]})

    course_vector = preprocessor.transform(df.loc[[idx], ['difficulty_level', 'course_price', 'feedback_score', 'course_duration_hours']])

    # Find neighbors
    distances, indices = knn_model.kneighbors(course_vector, n_neighbors=top_n*5)
    content_sim = 1 - distances.flatten()
    neighbor_ids = indices.flatten()

    candidates = df.iloc[neighbor_ids].copy()
    candidates['content_score'] = content_sim
    candidates['hybrid_score'] = w1 * candidates['content_score'] + w2 * candidates['popularity_score']

    return candidates.sort_values(by='hybrid_score', ascending=False).head(top_n)[
        ['course_id', 'course_name', 'difficulty_level', 'course_price', 'feedback_score', 'hybrid_score']
    ]

# Streamlit UI
st.title(" Hybrid Course Recommendation System")
course_list = df['course_name'].tolist()
selected_course = st.selectbox("Select a course:", course_list)
if st.button("Recommend"):
    recommendations = hybrid_recommend(selected_course, top_n=5)
    st.dataframe(recommendations)
