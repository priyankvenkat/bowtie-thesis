import os
from flask import send_from_directory
import dash
import dash_bootstrap_components as dbc

from layout import layout
from callbacks import register_callbacks
from callback_manual_match import register_manual_match_callbacks
from callback_review_tab import register_review_tab_callbacks

"""
Main entry point for the Dash app.

- Initializes the Dash application with styling and layout.
- Registers all callback modules (evaluation, manual override, review).
- Serves the final Excel file for download after metrics computation.
"""

# Initialize app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.title = "Bowtie GED Evaluator"
app.layout = layout

# Register callbacks
register_callbacks(app)
register_manual_match_callbacks(app)
register_review_tab_callbacks(app)
# register_node_override_callbacks(app)  # Enable if you want role override

# Ensure download folder exists
if not os.path.exists("downloads"):
    os.makedirs("downloads")

# Flask route to serve downloadable Excel file
@app.server.route("/download/<filename>")
def download_file(filename):
    return send_from_directory("downloads", filename, as_attachment=True)

# Run server
if __name__ == "__main__":
    print("✅ Callback functions loaded:")
    print("\n".join(app.callback_map.keys()))
    app.run(debug=True, port=8051)
