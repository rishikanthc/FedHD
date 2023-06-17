import hvplot.pandas
import holoviews as hv
from holoviews import opts
from bokeh.themes.theme import Theme
import pandas as pd
import os

hv.extension("bokeh")  # pyright:ignore
opts.defaults(
    opts.Curve(
        height=550,
        # height=300,
        # width=400,
        line_width=2.50,
        tools=["hover"],
        show_grid=True,
        # gridstyle=dict(grid_line_dash="dotted", grid_line_alpha=1.0),
    )
)

custom_theme = Theme(
    json={
        "attrs": {
            "Axes": {
                "axis_line_width": 10,
                "axis_label_text_font_size": "18px",
                "grid_line_color": "grey",
            },
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
                "background_fill_alpha": 0.8,
                "border_line_color": "black",
                "border_line_width": 2,
                "border_line_alpha": 0.8,
            },
            "Title": {
                "text_font_size": "24px",
                "text_font_style": "bold",
                "align": "center",
            },
        }
    }
)

hv.renderer("bokeh").theme = custom_theme


def genPlot(ds, root):
    # expt = "isolet"
    logfile1 = os.path.join(root, "logs", f"{ds}_hd.csv")
    logfile2 = os.path.join(root, "logs", f"{ds}_hd5.csv")
    logfile3 = os.path.join(root, "logs", f"{ds}_nn.csv")
    logfile4 = os.path.join(root, "logs", f"{ds}_nn5.csv")

    # df1 = pd.read_csv(logfile1, names=["Communication round", "Accuracy"], header=0)
    # df2 = pd.read_csv(logfile2, names=["Communication round", "Accuracy"], header=0)
    # df3 = pd.read_csv(logfile3, names=["Communication round", "Accuracy"], header=0)
    # df4 = pd.read_csv(logfile4, names=["Communication round", "Accuracy"], header=0)

    # df1 = pd.read_csv(logfile1, names=["Communication round", "E=1"], header=0)
    # df2 = pd.read_csv(logfile2, names=["Communication round", "E=5"], header=0)
    # df3 = pd.read_csv(logfile3, names=["Communication round", "E=1"], header=0)
    # df4 = pd.read_csv(logfile4, names=["Communication round", "E=5"], header=0)

    hd1 = pd.read_csv(logfile1)
    hd5 = pd.read_csv(logfile2)
    nn1 = pd.read_csv(logfile3)
    nn5 = pd.read_csv(logfile4)

    hdp1 = hd1.hvplot.line(
        x="Communication Round",
        y="Accuracy",
        color="#1f77b4",  # Blue
        line_width=2.5,
        label="FedHDC E=1",
    )

    hdp5 = hd5.hvplot.line(
        x="Communication Round",
        y="Accuracy",
        color="#ff7f0e",  # Orange
        line_width=2.5,
        label="FedHDC E=5",
    )

    nnp1 = nn1.hvplot.line(
        x="Communication Round",
        y="Accuracy",
        color="#1f77b4",  # Blue
        line_dash="dotted",
        line_width=3.0,
        label="NN E=1",
    )

    nnp5 = nn5.hvplot.line(
        x="Communication Round",
        y="Accuracy",
        color="#ff7f0e",  # Orange
        line_dash="dotted",
        line_width=3.0,
        label="NN E=5",
    )

    combo = (hdp1 * hdp5 * nnp1 * nnp5).opts(
        legend_position="bottom_right", title=ds.upper()
    )

    # hd1["Model"] = "FedHD"
    # hd1["Config"] = "E = 1"
    # hd5["Model"] = "FedHD"
    # hd5["Config"] = "E = 5"

    # nn1["Model"] = "NN"
    # nn1["Config"] = "E = 1"
    # nn5["Model"] = "NN"
    # nn5["Config"] = "E = 5"

    # df_all = pd.concat([hd1, hd5, nn1, nn5], ignore_index=True)
    # combo = df_all.hvplot.line(
    #     x="Communication Round", y="Accuracy", by=["Model", "Config"], legend="top"
    # )

    return combo


root = "/home/the-noetic/cookiejar/FedHD/"
isolet_plt = genPlot("isolet", root)
ucihar_plt = genPlot("ucihar", root)
face_plt = genPlot("face", root)
pamap_plt = genPlot("pamap", root)

plt = isolet_plt + ucihar_plt + face_plt + pamap_plt

saveisolet = os.path.join(root, "figs", f"combo.png")
hvplot.save(plt, saveisolet)
