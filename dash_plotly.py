import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

pd.set_option('display.max_columns', None)


class Config:
    SRC_PATH = Path("datasets/superstore.csv")

    COLOR_GOOD = "#27ae60"
    COLOR_BAD = "#e74c3c"
    COLOR_NEUTRAL = "#95a5a6"
    COLOR_SPARK = "#2c3e50"       # sparkline line color, matches KPI value text
    COLOR_CURRENT = "#2c3e50"     # "current period" bar color
    COLOR_PREVIOUS = "#95a5a6"    # "previous period" bar color


class DataManager:
    def __init__(self, path: Path):
        self.path = path
        self.df = self._load_data(path)

        self.region_options = sorted(self.df['region'].dropna().unique())
        self.category_options = sorted(self.df['category'].dropna().unique())
        self.subcat_options = sorted(self.df['sub_category'].dropna().unique())

        # Fixed color per category so it stays consistent across all charts
        palette = px.colors.qualitative.Set2
        self.category_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(self.category_options)}

        self.min_date = self.df['order_date'].min()
        self.max_date = self.df['order_date'].max()

    @staticmethod
    def _load_data(path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(f"There is no processed CSV file at: {path}")
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        date_cols = df.columns[df.columns.str.endswith('date')]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
        df['order_month'] = df['order_date'].dt.to_period('M').dt.to_timestamp()
        df['order_year'] = df['order_date'].dt.year
        df['original_sales'] = df['sales'] / (1 - df['discount'])
        return df

    @staticmethod
    def calc_growth(current, previous):
        if previous in (0, None) or (isinstance(previous, float) and pd.isna(previous)):
            return None
        return (current - previous) / previous * 100

    def apply_filters(self, region, category, subcat, start_date, end_date, base_df=None):
        out = self.df if base_df is None else base_df
        if region:
            out = out[out['region'] == region]
        if category:
            out = out[out['category'] == category]
        if subcat:
            out = out[out['sub_category'] == subcat]
        if start_date is not None and end_date is not None:
            out = out[(out['order_date'] >= pd.to_datetime(start_date)) &
                      (out['order_date'] <= pd.to_datetime(end_date))]
        return out

    @staticmethod
    def previous_period_range(start_date, end_date):
        """Previous period = an equal-length window immediately before the selected range."""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        span = end_date - start_date
        prev_end = start_date - pd.Timedelta(days=1)
        prev_start = prev_end - span
        return prev_start, prev_end


class Formatter:
    @staticmethod
    def fmt_money(v):
        if v >= 1_000_000:
            return f"${v/1_000_000:,.2f}M"
        if v >= 1_000:
            return f"${v/1_000:,.1f}K"
        return f"${v:,.0f}"

    @staticmethod
    def hex_to_rgba(hex_color, alpha):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    @classmethod
    def growth_span(cls, current, previous, positive_is_good=True, suffix="vs previous period"):
        growth = DataManager.calc_growth(current, previous)
        if growth is None:
            # No previous-period value to compare against: leave this line
            # blank instead of showing a placeholder message.
            return ""
        arrow = "▲" if growth >= 0 else "▼"
        is_good = (growth >= 0) == positive_is_good
        color_class = "text-success-custom" if is_good else "text-danger-custom"
        return html.Span(f"{arrow} {abs(growth):.1f}% {suffix}", className=color_class)


class ChartBuilder:
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    @staticmethod
    def make_sparkline(values, color=Config.COLOR_SPARK):
        """Tiny trend line for a KPI card: no axes, no grid, just the shape of
        the last few months. Height is fixed at 28px to match the KPI card.
        The shaded area is filled down to the series' own minimum (not a
        fixed y=0 baseline), so it always hugs the bottom of the line even
        when values dip below zero or never get close to zero."""
        if values:
            baseline = min(values)
            pad = (max(values) - baseline) * 0.05 or 1
            baseline -= pad
        else:
            baseline = 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[baseline] * len(values), mode='lines',
            line=dict(width=0), hoverinfo='skip', showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            y=values, mode='lines', line=dict(width=1.6, color=color),
            fill='tonexty', fillcolor=Formatter.hex_to_rgba(color, 0.12),
            showlegend=False,
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0), height=28,
            xaxis=dict(visible=False, fixedrange=True),
            yaxis=dict(visible=False, fixedrange=True),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
        )
        return fig

    @staticmethod
    def empty_figure():
        fig = go.Figure()
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        return fig

    def sales_vs_profit_chart(self, current_df, category_selected, target_margin=None):
        if current_df.empty:
            return self.empty_figure()

        scatter_col = 'sub_category' if category_selected else 'category'
        scatter_data = current_df.groupby(scatter_col).agg(
            sales=('sales', 'sum'), profit=('profit', 'sum'),
            orders=('order_id', 'nunique') if 'order_id' in current_df.columns else ('sales', 'count'),
        ).reset_index()
        scatter_data['margin'] = np.where(scatter_data['sales'] > 0,
                                           scatter_data['profit'] / scatter_data['sales'] * 100, 0)

        overall_margin = (current_df['profit'].sum() / current_df['sales'].sum() * 100
                           if current_df['sales'].sum() else 0)

        fig = px.scatter(
            scatter_data, x='sales', y='margin', size='orders', text=scatter_col,
            color='margin', color_continuous_scale=[Config.COLOR_BAD, Config.COLOR_NEUTRAL, Config.COLOR_GOOD],
            range_color=[min(0, scatter_data['margin'].min()),
                         max(overall_margin * 1.5, scatter_data['margin'].max())],
            labels={'sales': 'Sales ($)', 'margin': 'Profit margin (%)'},
        )
        fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='white')))

        # Give the point labels (drawn above each marker) enough head-room
        # so a label near the top of the chart (e.g. "Office Supplies")
        # doesn't get clipped by the plot's top edge.
        margin_min = scatter_data['margin'].min()
        margin_max = scatter_data['margin'].max()
        if target_margin is not None:
            # Make sure the target line/zone stays inside the visible range.
            margin_min = min(margin_min, target_margin)
            margin_max = max(margin_max, target_margin)
        margin_span = max(margin_max - margin_min, 1)
        fig.update_yaxes(range=[margin_min - margin_span * 0.15, margin_max + margin_span * 0.30],
                         showgrid=True, gridcolor='rgba(0,0,0,.22)', gridwidth=1, zeroline=False)
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,.22)', gridwidth=1, zeroline=False)

        # Target-margin line + red "below target" zone - only drawn once the
        # user enters a target profit margin in the Filters panel.
        if target_margin is not None:
            x_min = 0
            x_max = scatter_data['sales'].max() * 1.15
            y_min = min(scatter_data['margin'].min(), target_margin, 0) - 3
            fig.add_shape(type='rect', x0=x_min, x1=x_max, y0=y_min, y1=target_margin,
                          fillcolor=Config.COLOR_BAD, opacity=0.28, line_width=0, layer='below')
            fig.add_hline(y=target_margin, line_dash='dash', line_color=Config.COLOR_BAD,
                         annotation_text=f'Target margin: {target_margin:.1f}%', annotation_font_size=9)

        fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), coloraxis_showscale=False,
                          plot_bgcolor='white', paper_bgcolor='white')
        return fig

    def donut_chart(self, current_df, category_selected, cur_sales):
        if current_df.empty or cur_sales <= 0:
            return self.empty_figure()

        donut_col = 'sub_category' if category_selected else 'category'
        donut_data = current_df.groupby(donut_col)['sales'].sum().reset_index().sort_values('sales', ascending=False)
        color_map = self.dm.category_colors if donut_col == 'category' else None

        fig = px.pie(donut_data, names=donut_col, values='sales', hole=0.62,
                     color=donut_col, color_discrete_map=color_map)
        fig.update_traces(textinfo='percent+label', textposition='outside')
        fig.update_layout(
            showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(text=f"Total Sales<br><b>{Formatter.fmt_money(cur_sales)}</b>", x=0.5, y=0.5,
                              font_size=13, showarrow=False)],
        )
        return fig

    def category_trend_chart(self, current_df, category_selected):
        if current_df.empty:
            return self.empty_figure()

        trend_col = 'sub_category' if category_selected else 'category'
        yearly = current_df.groupby(['order_year', trend_col])['sales'].sum().reset_index()
        color_map = self.dm.category_colors if trend_col == 'category' else None

        fig = px.bar(yearly, x='order_year', y='sales', color=trend_col, barmode='stack',
                     color_discrete_map=color_map,
                     labels={'order_year': 'Year', 'sales': 'Sales ($)', trend_col: 'Category'})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation='h', y=-0.25),
                          plot_bgcolor='white', paper_bgcolor='white')
        fig.update_xaxes(type='category')
        return fig

    def monthly_trend_chart(self, current_df, monthly_cat):
        monthly_df = current_df if not monthly_cat else current_df[current_df['category'] == monthly_cat]
        if monthly_df.empty:
            return self.empty_figure()

        monthly = monthly_df.groupby('order_month')['sales'].sum().reset_index().sort_values('order_month')

        n_months = len(monthly)
        step = max(1, -(-n_months // 6))  # ~6 ticks max
        tick_vals = monthly['order_month'].iloc[::step]

        fig = go.Figure()
        # A single line drawn through the actual monthly sales points (no
        # smoothing/moving-average line) - markers sit right on the line.
        fig.add_trace(go.Scatter(x=monthly['order_month'], y=monthly['sales'], mode='lines+markers',
                                 line=dict(width=2.5, color='#2c3e50'),
                                 marker=dict(size=6, color='#2c3e50'), name='Monthly sales'))
        fig.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
                          margin=dict(l=10, r=10, t=10, b=10))
        fig.update_xaxes(tickvals=tick_vals, tickformat='%b<br>%Y', tickangle=0,
                         tickfont=dict(size=10), automargin=True)
        return fig


class LayoutBuilder:
    INDEX_STRING = '''
        <!DOCTYPE html>
        <html lang="en" dir="ltr">
        <head>
        {%metas%}
        <title>Management Dashboard</title>
        {%css%}
        <style>
        body{background-color:#0000;font-family:Tahoma,Arial,sans-serif;margin:0;padding:0}
        .dashboard-title{font-size:20px;font-weight:800;color:#2c3e50;margin:0;line-height:1.2;padding:2px 0 6px 0;text-align:left}
        .dashboard-subtitle{font-size:.75rem;color:#7f8c8d;margin-bottom:4px}
        .kpi-card{min-height:56px;text-align:center;padding:4px 6px;border-radius:7px;background:#eaf4fc;border:1px solid #a9cdea;box-shadow:0 3px 8px rgba(52,120,190,.18);display:flex;flex-direction:column;justify-content:center;margin-bottom:6px}
        /* Title (left) + value (right) on one row inside the KPI card. */
        .kpi-header{display:flex;flex-direction:row;justify-content:space-between;align-items:baseline;width:100%}
        .kpi-title{font-size:.82rem;color:#2c3e50;margin-bottom:0;font-weight:800;text-align:left}
        .kpi-value{font-size:1.05rem;font-weight:bold;color:#2c3e50;line-height:1.1;text-align:right}
        .kpi-sub{font-size:.6rem;margin-top:1px;line-height:1.3;min-height:.75rem}
        .text-success-custom{color:#27ae60!important;font-weight:600}
        .text-danger-custom{color:#e74c3c!important;font-weight:600}
        .text-muted-custom{color:#95a5a6!important}
        .chart-card{background:#fff;padding:8px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.06);height:100%}
        .chart-card h5{font-size:.85rem!important;margin:2px 0!important;font-weight:700}
        .summary-box{background:#eef6f9;padding:6px 10px;border-radius:6px;margin-bottom:4px;font-size:.72rem;line-height:1.7}
        .summary-box b{color:#2c3e50}
        .filter-button{margin-top:0;margin-bottom:7px;width:100%;font-weight:bold;font-size:.8rem;padding:6px}
        .filter-label{font-weight:bold;font-size:.85rem;color:#2c3e50;margin-bottom:5px}
        .compare-label{font-size:.62rem;color:#7f8c8d;text-align:center;margin-top:3px;margin-bottom:4px}
        .quick-range-btn{font-size:.66rem!important;padding:3px 4px!important}
        .alert-item{font-size:.75rem;line-height:1.7;margin-bottom:2px}
        .alert-item.good{color:#1e7e42}
        .alert-item.bad{color:#c0392b}
        .alert-item.neutral{color:#555}
        .no-alerts{font-size:.75rem;color:#7f8c8d}
        .DateInput_input{font-size:.8rem!important}
        .dash-graph{margin-bottom:0!important}
        @media(max-width:992px){.kpi-card{min-height:52px}.kpi-value{font-size:1rem}.dashboard-title{font-size:19px}}
        </style>
        </head>
        <body>
        {%app_entry%}
        <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
        </footer>
        </body>
        </html>
        '''

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    def _kpi_card(self, title, value_id, growth_id, spark_id, extra_children=None):
        children = [
            html.Div([
                html.Div(title, className='kpi-title'),
                html.Div(id=value_id, className='kpi-value'),
            ], className='kpi-header'),
            html.Div(id=growth_id, className='kpi-sub'),
        ]
        if extra_children:
            children.extend(extra_children)
        children.append(dcc.Graph(id=spark_id, config={'displayModeBar': False}, style={'height': '28px'}))
        return html.Div(children, className='kpi-card')

    def build_filter_sidebar(self):
        return dbc.Offcanvas(
            [
                html.H4("Filters", className="mb-4",
                        style={"fontWeight": "bold", "textAlign": "center", "color": "#2c3e50"}),
                html.Label("Region", className="filter-label"),
                dcc.Dropdown(id='region-filter',
                             options=[{'label': r, 'value': r} for r in self.dm.region_options],
                             placeholder='All regions', clearable=True, className='mb-3'),
                html.Label("Category", className="filter-label"),
                dcc.Dropdown(id='cat-filter',
                             options=[{'label': c, 'value': c} for c in self.dm.category_options],
                             placeholder='All categories', clearable=True, className='mb-3'),
                html.Label("Date Range", className="filter-label"),
                dcc.DatePickerRange(id='date-range', start_date=self.dm.min_date, end_date=self.dm.max_date,
                                    display_format='YYYY-MM-DD', className='mb-3'),
                html.Label("Target Profit Margin (%)", className="filter-label"),
                dcc.Input(id='target-margin-input', type='number', placeholder='e.g. 15',
                          debounce=True, className='mb-3 form-control'),
                # Hidden until a category is picked in the dropdown above -
                # toggled by the callback below.
                html.Div(
                    [
                        html.Hr(),
                        dbc.Accordion(
                            [
                                dbc.AccordionItem(
                                    [dcc.Dropdown(id='subcat-filter',
                                                  options=[{'label': s, 'value': s} for s in self.dm.subcat_options],
                                                  placeholder='All sub-categories', clearable=True)],
                                    title="Advanced filter: Sub-category",
                                )
                            ],
                            start_collapsed=True,
                        ),
                    ],
                    id='subcat-advanced-wrapper', style={'display': 'none'},
                ),
            ],
            id="filter-sidebar", title="Filters", is_open=False, placement="start",
            backdrop=True, scrollable=True, style={"width": "320px"},
        )

    def build_quick_range_buttons(self):
        quick_buttons = dbc.ButtonGroup([
            dbc.Button("30D", id='btn-range-30d', size='sm', color='light', outline=True, className='quick-range-btn'),
            dbc.Button("Year", id='btn-range-year', size='sm', color='light', outline=True, className='quick-range-btn'),
            dbc.Button("All", id='btn-range-all', size='sm', color='light', outline=True, className='quick-range-btn'),
        ], className='mb-2', style={'width': '100%'})

        # Custom range: pick any amount of days/weeks/months back from the
        # latest date in the data, e.g. "2 weeks" or "3 months".
        custom_range = dbc.InputGroup([
            dbc.Input(id='custom-range-value', type='number', min=1, step=1, placeholder='e.g. 2', size='sm'),
            dbc.Select(id='custom-range-unit', size='sm', value='days', options=[
                {'label': 'Days', 'value': 'days'},
                {'label': 'Weeks', 'value': 'weeks'},
                {'label': 'Months', 'value': 'months'},
            ]),
            dbc.Button('Set', id='btn-custom-range', size='sm', color='primary'),
        ], size='sm', className='mb-2')

        return html.Div([quick_buttons, custom_range])

    def build_kpi_row(self):
        return html.Div([
            self._kpi_card('Total Sales', 'kpi-sales', 'kpi-sales-growth', 'kpi-sales-spark'),
            self._kpi_card('Total Orders', 'kpi-orders', 'kpi-orders-growth', 'kpi-orders-spark'),
            self._kpi_card('Avg Order Value', 'kpi-aov', 'kpi-aov-growth', 'kpi-aov-spark'),
            self._kpi_card('Profit Margin', 'kpi-margin', 'kpi-margin-growth', 'kpi-margin-spark'),
            html.Div(id='compare-period-label', className='compare-label'),
        ])

    def build_period_highlights_box(self):
        return html.Div([
            html.Div('Period Highlights', className='kpi-title', style={'marginBottom': '4px'}),
            html.Div(id='highlights-body', style={'fontSize': '.72rem', 'lineHeight': '1.9'}),
        ], className='kpi-card', style={'textAlign': 'left', 'alignItems': 'flex-start'})

    def build_alerts_popup(self):
        return dbc.Toast(
            id='alerts-toast', header="⚠ Key Alerts", icon="warning", is_open=False, dismissable=True,
            style={"position": "fixed", "top": 12, "right": 12, "width": 340, "zIndex": 1050,
                   "boxShadow": "0 4px 14px rgba(0,0,0,.15)"},
        )

    def build_chart_rows(self):
        row1 = dbc.Row([
            dbc.Col([html.H5("Sales vs Profit", style={'textAlign': 'center'}),
                     dcc.Graph(id='sales-vs-profit-chart', config={'displayModeBar': False}, style={'height': '310px'})],
                    className='chart-card', md=6),
            dbc.Col([html.H5("Category Share of Sales", style={'textAlign': 'center'}),
                     dcc.Graph(id='donut-chart', config={'displayModeBar': False}, style={'height': '310px'})],
                    className='chart-card', md=6),
        ], className='g-2 mb-2')

        row2 = dbc.Row([
            dbc.Col([html.H5("Annual Sales by Category", style={'textAlign': 'center', 'marginBottom': '3px'}),
                     dcc.Graph(id="category-trend-chart", config={'displayModeBar': False}, style={'height': '340px'})],
                    className="chart-card", md=6),
            dbc.Col([html.H5("Monthly Sales Trend", style={'textAlign': 'center', 'marginBottom': '3px'}),
                     dcc.Dropdown(id='monthly-cat-filter',
                                  options=[{'label': c, 'value': c} for c in self.dm.category_options],
                                  placeholder='Show a specific category (optional)', clearable=True,
                                  style={'marginBottom': '4px'}),
                     dcc.Graph(id='monthly-trend-chart', config={'displayModeBar': False}, style={'height': '300px'})],
                    className="chart-card", md=6),
        ], className='g-2')

        return row1, row2

    def build_layout(self):
        filter_button = dbc.Button("☰ Filters", id="open-filter", color="primary", className="filter-button")
        quick_range_buttons = self.build_quick_range_buttons()
        kpi_row = self.build_kpi_row()
        period_highlights_box = self.build_period_highlights_box()
        chart_row1, chart_row2 = self.build_chart_rows()

        return dbc.Container([
            html.Div("Management Dashboard", className='dashboard-title'),
            dbc.Row([
                dbc.Col([filter_button, quick_range_buttons, kpi_row, period_highlights_box],
                        md=2, className='pe-1 order-1', style={'order': 1, 'paddingTop': '18px'}),
                dbc.Col([chart_row1, chart_row2], md=10, className='ps-1 order-2', style={'order': 2}),
            ], className='g-2', style={'display': 'flex', 'flexDirection': 'row'}),
            self.build_filter_sidebar(),
            self.build_alerts_popup(),
        ], fluid=True, style={'padding': '2px 10px 4px 10px'})


class DashboardApp:
    def __init__(self, src_path: Path = Config.SRC_PATH):
        self.dm = DataManager(src_path)
        self.charts = ChartBuilder(self.dm)
        self.layout_builder = LayoutBuilder(self.dm)

        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.index_string = LayoutBuilder.INDEX_STRING
        self.app.layout = self.layout_builder.build_layout()

        self._register_callbacks()

    # callbacks
    def _register_callbacks(self):
        app = self.app
        dm = self.dm
        charts = self.charts

        @app.callback(Output("filter-sidebar", "is_open"), Input("open-filter", "n_clicks"),
                      prevent_initial_call=True)
        def toggle_filter(n_clicks):
            return True

        @app.callback(Output('subcat-advanced-wrapper', 'style'), Input('cat-filter', 'value'))
        def toggle_subcat_filter(category):
            # Only show the "Advanced filter: Sub-category" accordion once a
            # category has been picked.
            return {'display': 'block'} if category else {'display': 'none'}

        @app.callback(
            Output('date-range', 'start_date'),
            Output('date-range', 'end_date'),
            Input('btn-range-30d', 'n_clicks'),
            Input('btn-range-year', 'n_clicks'),
            Input('btn-range-all', 'n_clicks'),
            Input('btn-custom-range', 'n_clicks'),
            State('custom-range-value', 'value'),
            State('custom-range-unit', 'value'),
            prevent_initial_call=True,
        )
        def set_quick_range(n_30d, n_year, n_all, n_custom, custom_value, custom_unit):
            clicked = ctx.triggered_id
            if clicked == 'btn-range-30d':
                end = dm.max_date
                start = end - pd.Timedelta(days=30)
                return start, end
            if clicked == 'btn-range-year':
                end = dm.max_date
                start = pd.Timestamp(year=end.year, month=1, day=1)
                return start, end
            if clicked == 'btn-range-all':
                return dm.min_date, dm.max_date
            if clicked == 'btn-custom-range':
                if not custom_value or custom_value <= 0:
                    return no_update, no_update
                end = dm.max_date
                if custom_unit == 'weeks':
                    start = end - pd.Timedelta(weeks=custom_value)
                elif custom_unit == 'months':
                    start = end - pd.DateOffset(months=custom_value)
                else:
                    start = end - pd.Timedelta(days=custom_value)
                start = max(start, dm.min_date)
                return start, end
            return dm.min_date, dm.max_date

        @app.callback(
            Output('kpi-sales', 'children'), Output('kpi-sales-growth', 'children'), Output('kpi-sales-spark', 'figure'),
            Output('kpi-orders', 'children'), Output('kpi-orders-growth', 'children'), Output('kpi-orders-spark', 'figure'),
            Output('kpi-aov', 'children'), Output('kpi-aov-growth', 'children'), Output('kpi-aov-spark', 'figure'),
            Output('kpi-margin', 'children'), Output('kpi-margin-growth', 'children'), Output('kpi-margin-spark', 'figure'),
            Output('highlights-body', 'children'),
            Output('compare-period-label', 'children'),
            Output('alerts-toast', 'children'), Output('alerts-toast', 'is_open'),
            Output('sales-vs-profit-chart', 'figure'),
            Output('donut-chart', 'figure'),
            Output('category-trend-chart', 'figure'),
            Output('monthly-trend-chart', 'figure'),
            Input('region-filter', 'value'),
            Input('cat-filter', 'value'),
            Input('subcat-filter', 'value'),
            Input('date-range', 'start_date'),
            Input('date-range', 'end_date'),
            Input('monthly-cat-filter', 'value'),
            Input('target-margin-input', 'value'),
        )
        def update_dashboard(region, category, subcat, start_date, end_date, monthly_cat, target_margin):
            return self._compute_dashboard(region, category, subcat, start_date, end_date, monthly_cat, target_margin)

    # main computation, kept as one method so all KPI/alert logic stays together 
    def _compute_dashboard(self, region, category, subcat, start_date, end_date, monthly_cat, target_margin=None):
        dm, charts = self.dm, self.charts

        start_date = start_date or dm.min_date
        end_date = end_date or dm.max_date
        current_df = dm.apply_filters(region, category, subcat, start_date, end_date)
        prev_start, prev_end = dm.previous_period_range(start_date, end_date)
        previous_df = dm.apply_filters(region, category, subcat, prev_start, prev_end)

        # KPIs 
        cur_sales = current_df['sales'].sum()
        prev_sales = previous_df['sales'].sum()
        cur_orders = current_df['order_id'].nunique() if 'order_id' in current_df.columns else len(current_df)
        prev_orders = previous_df['order_id'].nunique() if 'order_id' in previous_df.columns else len(previous_df)
        cur_aov = cur_sales / cur_orders if cur_orders else 0
        prev_aov = prev_sales / prev_orders if prev_orders else 0
        cur_profit = current_df['profit'].sum()
        prev_profit = previous_df['profit'].sum()
        cur_margin = (cur_profit / cur_sales * 100) if cur_sales else 0
        prev_margin = (prev_profit / prev_sales * 100) if prev_sales else 0

        kpi_sales_val = Formatter.fmt_money(cur_sales)
        kpi_sales_growth = Formatter.growth_span(cur_sales, prev_sales, positive_is_good=True)
        kpi_orders_val = f"{cur_orders:,}"
        kpi_orders_growth = Formatter.growth_span(cur_orders, prev_orders, positive_is_good=True)
        kpi_aov_val = f"${cur_aov:,.0f}"
        kpi_aov_growth = Formatter.growth_span(cur_aov, prev_aov, positive_is_good=True)
        kpi_margin_val = f"{cur_margin:.1f}%"
        kpi_margin_growth = Formatter.growth_span(cur_margin, prev_margin, positive_is_good=True)

        compare_label = f"{prev_start.date()} – {prev_end.date()}"

        # Sparklines
        if not current_df.empty:
            monthly_all = current_df.groupby('order_month').agg(
                sales=('sales', 'sum'),
                orders=('order_id', 'nunique') if 'order_id' in current_df.columns else ('sales', 'count'),
                profit=('profit', 'sum'),
            ).reset_index().sort_values('order_month')
            monthly_all['aov'] = np.where(monthly_all['orders'] > 0, monthly_all['sales'] / monthly_all['orders'], 0)
            monthly_all['margin'] = np.where(monthly_all['sales'] > 0,
                                              monthly_all['profit'] / monthly_all['sales'] * 100, 0)
        else:
            monthly_all = pd.DataFrame(columns=['order_month', 'sales', 'orders', 'profit', 'aov', 'margin'])

        fig_spark_sales = charts.make_sparkline(monthly_all['sales'].tolist())
        fig_spark_orders = charts.make_sparkline(monthly_all['orders'].tolist())
        fig_spark_aov = charts.make_sparkline(monthly_all['aov'].tolist())
        fig_spark_margin = charts.make_sparkline(monthly_all['margin'].tolist())

        # Period highlights
        if not current_df.empty:
            subcat_sales = current_df.groupby('sub_category')['sales'].sum().sort_values(ascending=False)
            top_subcat_txt = (f"{subcat_sales.index[0]} ({Formatter.fmt_money(subcat_sales.iloc[0])})"
                               if len(subcat_sales) else "—")
            month_sales = current_df.groupby('order_month')['sales'].sum().sort_values(ascending=False)
            busiest_month_txt = (f"{month_sales.index[0].strftime('%b %Y')} ({Formatter.fmt_money(month_sales.iloc[0])})"
                                  if len(month_sales) else "—")
            biggest_order_txt = Formatter.fmt_money(current_df['sales'].max())

            cat_profit = current_df.groupby('category')['profit'].sum().sort_values(ascending=False)
            top_profit_cat_txt = cat_profit.index[0] if len(cat_profit) else "—"

            region_sales = current_df.groupby('region')['sales'].sum().sort_values(ascending=False)
            top_region_txt = (f"{region_sales.index[0]} ({Formatter.fmt_money(region_sales.iloc[0])})"
                               if len(region_sales) else "—")

            avg_discount_txt = f"{current_df['discount'].mean() * 100:.1f}%"
        else:
            top_subcat_txt = busiest_month_txt = biggest_order_txt = "—"
            top_profit_cat_txt = top_region_txt = avg_discount_txt = "—"

        highlights_children = html.Div([
            html.Div([html.B("Top sub-category: "), top_subcat_txt]),
            html.Div([html.B("Busiest month: "), busiest_month_txt]),
            html.Div([html.B("Biggest order: "), biggest_order_txt]),
            html.Div([html.B("Most profitable category: "), top_profit_cat_txt]),
            html.Div([html.B("Top region: "), top_region_txt]),
            html.Div([html.B("Avg discount: "), avg_discount_txt]),
        ])

        # Alerts
        alerts = []
        margin_growth = dm.calc_growth(cur_margin, prev_margin)
        if margin_growth is not None and cur_margin - prev_margin <= -1:
            alerts.append(('bad', f"Profit margin dropped {abs(cur_margin - prev_margin):.1f} points."))
        elif margin_growth is not None and cur_margin - prev_margin >= 1:
            alerts.append(('good', f"Profit margin improved {cur_margin - prev_margin:.1f} points."))

        aov_growth = dm.calc_growth(cur_aov, prev_aov)
        sales_growth = dm.calc_growth(cur_sales, prev_sales)
        if sales_growth is not None and sales_growth > 5 and aov_growth is not None and aov_growth < sales_growth / 2:
            alerts.append(('neutral', f"Sales growth ({sales_growth:.0f}%) is mainly driven by more orders, "
                                       f"not a higher order value."))

        cat_group_col = 'sub_category' if category else 'category'
        if not current_df.empty:
            cur_cat = current_df.groupby(cat_group_col).agg(sales=('sales', 'sum'), profit=('profit', 'sum')).reset_index()
            prev_cat = (previous_df.groupby(cat_group_col).agg(sales=('sales', 'sum')).reset_index()
                        .rename(columns={'sales': 'prev_sales'}))
            merged_cat = cur_cat.merge(prev_cat, on=cat_group_col, how='left')
            merged_cat['margin'] = np.where(merged_cat['sales'] > 0,
                                             merged_cat['profit'] / merged_cat['sales'] * 100, 0)
            merged_cat['growth'] = merged_cat.apply(lambda r: dm.calc_growth(r['sales'], r['prev_sales']), axis=1)

            # Low-profitability threshold is now relative to the period's own
            # overall margin (instead of the removed fixed TARGET_MARGIN).
            low_profit_threshold = target_margin if target_margin is not None else cur_margin * 0.5

            for _, row in merged_cat.iterrows():
                if row['growth'] is not None and row['growth'] > 20:
                    alerts.append(('good', f"Sales of \"{row[cat_group_col]}\" grew {row['growth']:.0f}%."))
                if row['margin'] < low_profit_threshold and row['sales'] > 0:
                    alerts.append(('bad', f"\"{row[cat_group_col]}\" has low profitability despite sales "
                                           f"({row['margin']:.1f}%)."))

        if not alerts:
            alerts_children = html.Div("No warning-level items right now.", className='no-alerts')
            alerts_is_open = False
        else:
            alerts_children = [html.Div(f"• {msg}", className=f'alert-item {kind}') for kind, msg in alerts[:6]]
            alerts_is_open = True

        # Charts
        fig_sp = charts.sales_vs_profit_chart(current_df, category, target_margin)
        fig_donut = charts.donut_chart(current_df, category, cur_sales)
        fig_cat_trend = charts.category_trend_chart(current_df, category)
        fig_month = charts.monthly_trend_chart(current_df, monthly_cat)

        return (
            kpi_sales_val, kpi_sales_growth, fig_spark_sales,
            kpi_orders_val, kpi_orders_growth, fig_spark_orders,
            kpi_aov_val, kpi_aov_growth, fig_spark_aov,
            kpi_margin_val, kpi_margin_growth, fig_spark_margin,
            highlights_children,
            compare_label,
            alerts_children, alerts_is_open,
            fig_sp, fig_donut, fig_cat_trend, fig_month,
        )

    def run(self, debug=True):
        self.app.run(debug=debug)


if __name__ == '__main__':
    dashboard = DashboardApp()
    dashboard.run(debug=True)