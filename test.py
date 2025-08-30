# %%
import plotly.io as pio
import plotly.graph_objects as go
# Either of these renderers work well in VS Code:
pio.renderers.default = "vscode"            # VS Code Jupyter/Interactive
# pio.renderers.default = "notebook_connected"  # also works

fig = go.Figure(data=go.Scatter(y=[1,3,2,4], mode="lines+markers"))
fig.update_layout(title="Hello Plotly (Remote)")
fig.show()


# %%
