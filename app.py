import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from src import stock_data, model_train, train_test_split, model_predict

st.image('images.jpg', caption='Image credit : Kelly Sikkema')
# Function to display result, history, and data information
def display_results(user_input,result, data_inf, train):
    st.header(f'Here is the data for {user_input.upper()} the past 10 days.')
    st.write(data_inf)
    plot = sns.lineplot(train)
    st.pyplot(plot.get_figure())
    future_date = datetime.now() + timedelta(days=1)
    formatted_date = future_date.strftime('%Y-%m-%d')
    st.markdown(f'''
        ## Stock Prediction Analysis for <span style="font-size:24px;">{user_input.upper()}</span>
        
        <p style="font-size:24px;">
        {formatted_date} {user_input.upper()} predicted price is : <b>{round(result,2)}</b>
        </p>
    ''', unsafe_allow_html=True)
    
    
    
    
# Main function to run the app
def main():
    st.title("Stock Prediction App")
    user_input = st.text_input("Enter your stock ticker here:")
    
    if st.button("Predict"):
        with st.spinner('Loading...'):
            train = stock_data(user_input)
            if len(train)>0: 
                x_train, y_train, scaler = train_test_split(train)
                history, model = model_train(x_train, y_train)
                result, data_inf = model_predict(train, scaler, model) 
                display_results(user_input,result[0][0], data_inf,train)
            else:
                st.write('Invalid stock ticker. Please verify the ticker symbol on the following website: [Yahoo Finance.](https://finance.yahoo.com/)')


if __name__ == "__main__":
    main()
