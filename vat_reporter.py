import pandas as pd
import numpy as np
from itertools import repeat

# from bokeh.io import output_file, show
from bokeh.layouts import column, row
from bokeh.models import Select, LinearAxis, Range1d
from bokeh.plotting import curdoc, figure

df = pd.read_csv('/home/patrick/code/video-annotation-tool/data/sample1/sample1.csv')
n_labels = df['labelled_frame'].iloc[-1]
time = [i+1 for i in range(n_labels)]

def _get_percent_on_patch(df, n_labels):
    percents = dict(zip(range(1,n_labels+1), repeat(None)))
    mini = df[['labelled_frame', 'on_patch']]
    for i in range(len(mini)):
        status = mini.iloc[i]['on_patch']
        label = mini.iloc[i]['labelled_frame']
        if percents[label] is not None:
            if status == True:
                percents[label].append(1)
            else:
                percents[label].append(0)
        else:
            percents[label] = [int(status)]
    for k in percents:
        percents[k] = np.mean(percents[k])
    return [percents[i+1] for i in range(n_labels)]

def _get_percent_of_behavior(df, n_labels, behavior):
    percents = dict(zip(range(1,n_labels+1), repeat(None)))
    mini = df[['labelled_frame', 'behavior']]
    for i in range(len(mini)):
        status = mini.iloc[i]['behavior']
        label = mini.iloc[i]['labelled_frame']
        if percents[label] is not None:
            if status == behavior:
                percents[label].append(1)
            else:
                percents[label].append(0)
        else:
            if status == behavior:
                percents[label] = [1]
            else:
                percents[label] = [0]
    for k in percents:
        percents[k] = np.mean(percents[k])
    return [percents[i+1] for i in range(n_labels)]

def create_figure():
    xs = None
    y1s = None
    y2s = None
    x_title = "Labelled Frames"

    y_1 = y1.value
    y_2 = y2.value

    if y_1=='Select Factor 1':
        y1s = None
        xs = None
    elif y_1 == '%_copulating':
        xs = time
        y1s = _get_percent_of_behavior(df, n_labels, "Copulation")
    elif y_1 == '%_courting':
        xs = time
        y1s = _get_percent_of_behavior(df, n_labels, "Courting")
    elif y_1 == '%_nothing':
        xs = time
        y1s = _get_percent_of_behavior(df, n_labels, "Nothing")
    elif y_1 == '%_on_patch':
        xs = time
        y1s = _get_percent_on_patch(df, n_labels)

    if y_2=='Select Factor 1':
        y2s = None
        xs = None
    elif y_2 == '%_copulating':
        xs = time
        y2s = _get_percent_of_behavior(df, n_labels, "Copulation")
    elif y_2 == '%_courting':
        xs = time
        y2s = _get_percent_of_behavior(df, n_labels, "Courting")
    elif y_2 == '%_nothing':
        xs = time
        y2s = _get_percent_of_behavior(df, n_labels, "Nothing")
    elif y_2 == '%_on_patch':
        xs = time
        y2s = _get_percent_on_patch(df, n_labels)


    if y2s is None:
        start_pt, end_pt = 0.0, 1.0
    else:
        start_pt, end_pt = 0.0, 1.0
        #start_pt, end_pt = 0, np.max(y2s)

    p = figure(plot_height=1000, plot_width=1800, tools='pan,box_zoom,hover,reset')#, **kw)
    # p.extra_y_ranges = {"second":Range1d(start=start_pt, end=end_pt)}
    # p.add_layout(LinearAxis(y_range_name="second", axis_label= y2.value.title()), 'right')

    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = None


    sz = 9
    c1 = "#31AADE"
    c2 = "#FF0000"


    p.circle(x=xs, y=y1s, color=c1, size=sz, line_color=None, alpha=0.6, hover_color='white', hover_alpha=0.5)
    p.line(x = xs, y=y1s, color=c1,line_width=2)

    p.circle(x=xs, y=y2s, color=c2, size=sz, line_color=None, alpha=0.6, hover_color='white', hover_alpha=0.5)#,y_range_name='second')
    p.line(x = xs, y=y2s, color=c2, line_width=2)#, y_range_name='second')

    return p

def update(attr, old, new):
    layout.children[1] = create_figure()


y1 = Select(title='Factor 1 (Blue)', options=['Select Factor 1','%_on_patch', '%_nothing', '%_courting', '%_copulating'])
y1.on_change('value', update)

y2 = Select(title='Factor 1 (Red)', options=['Select Factor 2','%_on_patch', '%_nothing', '%_courting', '%_copulating'])
y2.on_change('value', update)

controls = column(y1, y2, width=200)
layout = row(controls, create_figure())

curdoc().add_root(layout)
curdoc().title = "Crossfilter"
