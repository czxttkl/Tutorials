import plotly.plotly as py
import plotly.graph_objs as go
import plotly.io as pio
import plotly

plotly.tools.set_credentials_file(username='czxttkl', api_key='z7x4u6A48PuOoNQs0lIx')


x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_rev = x[::-1]

# Line 1
y1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1_upper = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y1_lower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y1_lower = y1_lower[::-1]

# Line 2
y2 = [5, 2.5, 5, 7.5, 5, 2.5, 7.5, 4.5, 5.5, 5]
y2_upper = [5.5, 3, 5.5, 8, 6, 3, 8, 5, 6, 5.5]
y2_lower = [4.5, 2, 4.4, 7, 4, 2, 7, 4, 5, 4.75]
y2_lower = y2_lower[::-1]

# Line 3
y3 = [10, 8, 6, 4, 2, 0, 2, 4, 2, 0]
y3_upper = [11, 9, 7, 5, 3, 1, 3, 5, 3, 1]
y3_lower = [9, 7, 5, 3, 1, -.5, 1, 3, 1, -1]
y3_lower = y3_lower[::-1]

trace1 = go.Scatter(
    x=x+x_rev,
    y=y1_upper+y1_lower,
    fill='tozerox',
    fillcolor='rgba(0,100,80,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=False,
    name='Fair',
)
trace2 = go.Scatter(
    x=x+x_rev,
    y=y2_upper+y2_lower,
    fill='tozerox',
    fillcolor='rgba(0,176,246,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Premium',
    showlegend=False,
)
trace3 = go.Scatter(
    x=x+x_rev,
    y=y3_upper+y3_lower,
    fill='tozerox',
    fillcolor='rgba(231,107,243,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=False,
    name='Fair',
)
trace4 = go.Scatter(
    x=x,
    y=y1,
    line=dict(color='rgb(0,100,80)'),
    mode='lines',
    name='Fair',
)
trace5 = go.Scatter(
    x=x,
    y=y2,
    line=dict(color='rgb(0,176,246)'),
    mode='lines',
    name='Premium',
)
trace6 = go.Scatter(
    x=x,
    y=y3,
    line=dict(color='rgb(231,107,243)'),
    mode='lines',
    name='Ideal',
)

data = [trace1, trace2, trace3, trace4, trace5, trace6]

layout = go.Layout(
    paper_bgcolor='rgb(255,255,255)',
    plot_bgcolor='rgb(229,229,229)',
    xaxis=dict(
        gridcolor='rgb(255,255,255)',
        range=[1,10],
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        zeroline=False
    ),
    yaxis=dict(
        gridcolor='rgb(255,255,255)',
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        zeroline=False
    ),
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename= 'shaded_lines')

pio.write_image(fig, 'images/fig1.png')