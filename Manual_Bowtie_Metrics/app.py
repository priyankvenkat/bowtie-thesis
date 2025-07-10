# Entry point of the Dash app
import dash
from layout import layout
from callbacks import register_callbacks

app = dash.Dash(__name__)
app.title = "Bowtie GED (Image vs Image)"
app.config.suppress_callback_exceptions = True

# Assign layout
app.layout = layout

# Register all callbacks from callbacks.py
register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)
