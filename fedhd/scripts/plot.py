import hvplot.pandas
import holoviews as hv
from holoviews import opts
from bokeh.themes.theme import Theme
import pandas as pd
import os

hv.extension("bokeh")  # pyright:ignore
opts.defaults(
    opts.Curve(
        height=400,
        # height=300,
        # width=400,
        line_width=2.50,
        tools=["hover"],
        show_grid=True,
    ),
    opts.Distribution(
        height=400,
        # width=400,
        line_width=2.50,
        tools=["hover"],
        show_grid=True,
    ),
)

custom_theme = Theme(
    json={
        "attrs": {
            "Figure": {},
            "Axes": {"axis_line_width": 10, "axis_label_text_font_size": "18px"},
            "Axis": {
                "axis_label_text_font_size": "18px",
                "minor_tick_line_width": 0,
                "major_label_text_font_size": "22px",
                "major_label_text_font_style": "bold",
            },
            "Legend": {
                "label_text_font_size": "18px",
                "title_text_font_size": "20px",
                "title_text_font_style": "bold",
            },
        }
    }
)

hv.renderer("bokeh").theme = custom_theme


def genPlot(ds, root):
    # expt = "isolet"
    logfile1 = os.path.join(root, "logs", f"{ds}.csv")
    logfile2 = os.path.join(root, "logs", f"{ds}5.csv")

    df1 = pd.read_csv(logfile1, names=["Communication round", "E=1"], header=0)
    df2 = pd.read_csv(logfile2, names=["Communication round", "E=5"], header=0)
    print(df1.head())
    print(df2.head())

    df = pd.merge(df1, df2, on="Communication round")
    df = pd.melt(
        df,
        id_vars=["Communication round"],
        value_vars=["E=1", "E=5"],
        var_name="config",
        value_name="Accuracy",
    )
    print(df.head())

    plt = df.hvplot.line(x="Communication round", y="Accuracy", by="config")

    return plt


root = "/home/the-noetic/cookiejar/FedHD/"
isolet_plt = genPlot("isolet", root)
ucihar_plt = genPlot("ucihar", root)
face_plt = genPlot("face", root)
pamap_plt = genPlot("pamap", root)

plt = (isolet_plt + ucihar_plt + face_plt + pamap_plt).opts(legend_position="top")

saveisolet = os.path.join(root, "figs", f"combo.png")
# hvplot.save(plt, saveisolet)
