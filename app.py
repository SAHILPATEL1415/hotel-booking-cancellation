import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨")
st.title("🏨 Hotel Booking Cancellation Predictor")
st.write("Fill in the booking details to predict cancellation risk.")

model = joblib.load('hotel_model_small.pkl')

lead_time = st.slider("Lead Time (days)", 0, 500, 50)
adults = st.number_input("Adults", 1, 10, 2)
children = st.number_input("Children", 0, 5, 0)
adr = st.slider("Average Daily Rate", 0, 500, 100)
week_nights = st.slider("Week Nights", 0, 20, 2)
weekend_nights = st.slider("Weekend Nights", 0, 10, 1)
prev_cancel = st.number_input("Previous Cancellations", 0, 20, 0)
special_req = st.slider("Special Requests", 0, 5, 1)

if st.button("🔮 Predict Cancellation"):
    features = list(model.feature_names_in_)
    row = {f: 0 for f in features}

    row['lead_time'] = lead_time
    row['adults'] = adults
    row['children'] = children
    row['adr'] = adr
    row['stays_in_week_nights'] = week_nights
    row['stays_in_weekend_nights'] = weekend_nights
    row['previous_cancellations'] = prev_cancel
    row['total_of_special_requests'] = special_req
    row['total_guests'] = adults + children
    row['total_nights'] = week_nights + weekend_nights
    row['total_cost'] = adr * (week_nights + weekend_nights)
    row['is_family'] = 1 if children > 0 else 0

    input_df = pd.DataFrame([row])[features]
    prob = model.predict_proba(input_df)[0][1]

    st.metric("Cancellation Probability", f"{prob*100:.1f}%")

    if prob > 0.6:
        st.error("⚠️ HIGH RISK — Consider overbooking strategy")
    elif prob > 0.3:
        st.warning("⚡ MEDIUM RISK — Send reminder email")
    else:
        st.success("✅ LOW RISK — Safe booking")
