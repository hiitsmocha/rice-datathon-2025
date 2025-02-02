import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyDUZVf2L1nDjQ2Z9iLM12EWNROQnjuX-qU")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize session memory
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.title("📊 AI-Powered CSV Query & Visualization App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read CSV file
    df = pd.read_csv(uploaded_file)
    st.write("### 📂 Uploaded CSV Data Preview:")
    st.dataframe(df.head())  # Show first few rows

    # Show basic statistics
    st.write("### 📊 Dataset Summary Statistics:")
    st.write(df.describe())

    # Visualization section
    st.write("## 📈 Data Visualization")

    # Let user choose column for visualization
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=["number"]).columns.tolist()

    if numeric_columns:
        col_to_plot = st.selectbox("Select a numeric column for histogram:", numeric_columns)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[col_to_plot], bins=30, kde=True, ax=ax)
        ax.set_title(f"Histogram of {col_to_plot}")
        st.pyplot(fig)

    if categorical_columns:
        col_to_plot = st.selectbox("Select a categorical column for bar chart:", categorical_columns)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        df[col_to_plot].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Bar Chart of {col_to_plot}")
        st.pyplot(fig)

    # User query input
    user_query = st.text_area("Ask a question about this data:")

    if st.button("Get Answer"):
        if user_query:
            # Convert CSV data to text format
            csv_text = df.to_csv(index=False)

            # Construct Gemini prompt
            prompt = f"""
            You are an AI assistant analyzing a dataset. The user has uploaded the following CSV file:

            {csv_text}

            The user asks: "{user_query}"

            Please analyze the data and provide an insightful response.
            """
            
            # Get response from Gemini
            response = model.generate_content(prompt)

            # Save query & response to session memory
            st.session_state["chat_history"].append({"query": user_query, "response": response.text})

            # Display AI response
            st.write("### 🤖 Gemini AI's Answer:")
            st.write(response.text)
        else:
            st.warning("Please enter a query before clicking 'Get Answer'.")

    # Show chat history
    if st.session_state["chat_history"]:
        st.write("## 📝 Chat History")
        for chat in st.session_state["chat_history"]:
            with st.expander(f"📌 {chat['query']}"):
                st.write(chat["response"])

    # Option to clear memory
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.success("Chat history cleared!")
