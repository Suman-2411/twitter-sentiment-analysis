import plotly.express as px
import pandas as pd

# plotly is better than matplotlib 
# because you can hover on the chart

def make_comparison_chart(df_vader, df_roberta):
    categories = ['negative', 'neutral', 'positive']
    
    vader_counts = df_vader['vader_sentiment'].value_counts()
    roberta_counts = df_roberta['roberta_sentiment'].value_counts()
    
    # vader sometimes misses neutral
    # so use .get() to avoid errors
    vader_vals = [vader_counts.get(c, 0) for c in categories]
    roberta_vals = [roberta_counts.get(c, 0) for c in categories]
    
    chart_df = pd.DataFrame({
        'Sentiment': categories * 2,
        'Count': vader_vals + roberta_vals,
        'Model': ['VADER']*3 + ['RoBERTa']*3
    })
    
    # side by side bar chart
    fig = px.bar(chart_df,
                 x='Sentiment',
                 y='Count',
                 color='Model',
                 barmode='group',
                 title='VADER vs RoBERTa',
                 color_discrete_map={
                     'VADER': '#1d9bf0',
                     'RoBERTa': '#ff4444'
                 })
    return fig


def make_pie_chart(df, model_name):
    # pie chart looks better for this
    if 'vader_sentiment' in df.columns:
        counts = df['vader_sentiment'].value_counts()
    else:
        counts = df['roberta_sentiment'].value_counts()
    
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title=f'{model_name} results',
        color_discrete_sequence=[
            '#ff4444', '#aaaaaa', '#00cc66'
        ]
    )
    return fig