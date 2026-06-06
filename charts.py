import plotly.express as px
import pandas as pd

def make_comparison_chart(df_vader, df_roberta):
    # Categories
    categories = ['negative', 'neutral', 'positive']
    
    # Count sentiments
    vader_counts = df_vader['vader_sentiment'].value_counts()
    roberta_counts = df_roberta['roberta_sentiment'].value_counts()
    
    # Get values
    vader_vals = [vader_counts.get(c, 0) for c in categories]
    roberta_vals = [roberta_counts.get(c, 0) for c in categories]
    
    # Create dataframe
    chart_df = pd.DataFrame({
        'Sentiment': categories * 2,
        'Count': vader_vals + roberta_vals,
        'Model': ['VADER']*3 + ['RoBERTa']*3
    })
    
    # Create bar chart
    fig = px.bar(chart_df,
                 x='Sentiment',
                 y='Count',
                 color='Model',
                 barmode='group',
                 title='VADER vs RoBERTa Comparison',
                 color_discrete_map={
                     'VADER': '#1d9bf0',
                     'RoBERTa': '#ff4444'
                 })
    return fig


def make_pie_chart(df, model_name):
    # Get correct column
    if 'vader_sentiment' in df.columns:
        counts = df['vader_sentiment'].value_counts()
    else:
        counts = df['roberta_sentiment'].value_counts()
    
    # Create pie chart
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title=f'{model_name} Distribution',
        color_discrete_sequence=['#ff4444','#aaaaaa','#00cc66']
    )
    return fig