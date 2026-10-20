# ------------------------------------------------------------
# TEC304 Advanced Programming - Assessment 3 Hackathon
# World Happiness Report Analyser
# ------------------------------------------------------------
# This is a desktop GUI application made with Python.
# It loads World Happiness Report datasets from 2015 to 2019,
# allows the user to select a year, analyse data, create charts,
# train a machine learning model, and predict happiness score.
# ------------------------------------------------------------

# Tkinter is used to build the graphical user interface.
import tkinter as tk
from tkinter import ttk, messagebox

# os is used to check whether the CSV files exist in the folder.
import os

# pandas is used for reading, cleaning, and analysing CSV data.
import pandas as pd

# matplotlib is used for creating charts.
import matplotlib.pyplot as plt

# scikit-learn is used for the machine learning model.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# DATASET SETTINGS
# ------------------------------------------------------------

# These are the default CSV files that must be in the same folder as main.py.
DEFAULT_FILES = ["2015.csv", "2016.csv", "2017.csv", "2018.csv", "2019.csv"]

# These are the main factors used to analyse and predict happiness score.
FEATURES = [
    "GDP",
    "Social Support",
    "Life Expectancy",
    "Freedom",
    "Generosity",
    "Corruption"
]

# Different years have slightly different column names.
# This dictionary changes them into one common format.
COLUMN_RENAME = {
    "Country or region": "Country",
    "Country": "Country",

    "Happiness Rank": "Rank",
    "Happiness.Rank": "Rank",
    "Overall rank": "Rank",

    "Happiness Score": "Score",
    "Happiness.Score": "Score",
    "Score": "Score",

    "Economy (GDP per Capita)": "GDP",
    "Economy..GDP.per.Capita.": "GDP",
    "GDP per capita": "GDP",

    "Family": "Social Support",
    "Social support": "Social Support",

    "Health (Life Expectancy)": "Life Expectancy",
    "Health..Life.Expectancy.": "Life Expectancy",
    "Healthy life expectancy": "Life Expectancy",

    "Freedom": "Freedom",
    "Freedom to make life choices": "Freedom",

    "Generosity": "Generosity",

    "Trust (Government Corruption)": "Corruption",
    "Trust..Government.Corruption.": "Corruption",
    "Perceptions of corruption": "Corruption"
}


# ------------------------------------------------------------
# DATA LOADING AND CLEANING
# ------------------------------------------------------------

def clean_dataset(df, year):
    """Clean one CSV file and make its columns consistent."""

    # Rename columns so all year files follow the same format.
    df = df.rename(columns=COLUMN_RENAME)

    # Add the year manually because each CSV file represents one year.
    df["Year"] = year

    # These two columns are essential for the application.
    if "Country" not in df.columns:
        raise ValueError("Country column is missing.")

    if "Score" not in df.columns:
        raise ValueError("Happiness Score column is missing.")

    # Only keep feature columns that are available in the current file.
    available_features = [col for col in FEATURES if col in df.columns]

    if len(available_features) < 2:
        raise ValueError("Not enough feature columns found.")

    # Keep only the useful columns for this application.
    keep_columns = ["Country", "Score"] + available_features + ["Year"]

    if "Rank" in df.columns:
        keep_columns.insert(1, "Rank")

    df = df[keep_columns].copy()

    # Convert score and feature columns into numbers.
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows that have missing or invalid numeric values.
    df = df.dropna()

    return df


def load_all_data():
    """Load and combine all default CSV files from 2015 to 2019."""

    all_data = []

    for file_name in DEFAULT_FILES:
        # Stop the program with a clear message if a file is missing.
        if not os.path.exists(file_name):
            raise FileNotFoundError(
                f"{file_name} is missing. Keep all CSV files in the same folder as main.py."
            )

        # Extract year from file name, for example 2015.csv becomes 2015.
        year = int(file_name.replace(".csv", ""))

        # Read and clean each yearly dataset.
        df = pd.read_csv(file_name)
        df = clean_dataset(df, year)
        all_data.append(df)

    # Combine all years into one dataset.
    return pd.concat(all_data, ignore_index=True)


# ------------------------------------------------------------
# MAIN APPLICATION CLASS
# ------------------------------------------------------------

class HappinessApp:

    def __init__(self, root):
        # Basic window setup.
        self.root = root
        self.root.title("World Happiness Report Analyser")
        self.root.geometry("1100x720")

        # Load all default datasets when the app starts.
        self.data = load_all_data()

        # These variables are used later for machine learning and prediction.
        self.model = None
        self.model_features = []
        self.model_year_used = ""

        # Build the interface and show initial data.
        self.build_gui()
        self.update_home()
        self.populate_table()

    # --------------------------------------------------------
    # GUI SETUP
    # --------------------------------------------------------

    def build_gui(self):
        """Create the main window, controls, and tabs."""

        # Main title.
        tk.Label(
            self.root,
            text="World Happiness Report Analyser",
            font=("Arial", 20, "bold")
        ).pack(pady=10)

        # Short subtitle to explain the app purpose.
        tk.Label(
            self.root,
            text="Data Analysis | Year Selection | Visualisation | Machine Learning",
            font=("Arial", 11)
        ).pack()

        # Top control area for year selection and model training.
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(control_frame, text="Select Year:").pack(side="left", padx=5)

        # User can choose all years or one specific year.
        self.year_choice = tk.StringVar(value="All Years")

        ttk.Combobox(
            control_frame,
            textvariable=self.year_choice,
            values=["All Years", "2015", "2016", "2017", "2018", "2019"],
            state="readonly",
            width=15
        ).pack(side="left", padx=5)

        # Button for training the machine learning model.
        tk.Button(
            control_frame,
            text="Train Model",
            command=self.train_model,
            width=18
        ).pack(side="left", padx=10)

        # Status line shows important information to the user.
        self.status = tk.StringVar(value=f"Loaded default dataset: {len(self.data)} rows | 2015–2019")

        tk.Label(
            control_frame,
            textvariable=self.status
        ).pack(side="left", padx=15)

        # Notebook creates tabs inside the app.
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        # Create each tab.
        self.home_tab = tk.Frame(self.notebook)
        self.analysis_tab = tk.Frame(self.notebook)
        self.chart_tab = tk.Frame(self.notebook)
        self.ml_tab = tk.Frame(self.notebook)
        self.predict_tab = tk.Frame(self.notebook)
        self.table_tab = tk.Frame(self.notebook)

        # Add tabs to the notebook.
        self.notebook.add(self.home_tab, text="Home")
        self.notebook.add(self.analysis_tab, text="Analysis")
        self.notebook.add(self.chart_tab, text="Charts")
        self.notebook.add(self.ml_tab, text="ML Results")
        self.notebook.add(self.predict_tab, text="Predict")
        self.notebook.add(self.table_tab, text="Data Table")

        # Build each tab content.
        self.build_home_tab()
        self.build_analysis_tab()
        self.build_chart_tab()
        self.build_ml_tab()
        self.build_predict_tab()
        self.build_table_tab()

    def get_selected_data(self):
        """Return data based on the selected year."""

        selected = self.year_choice.get()

        if selected == "All Years":
            return self.data, "All Years"

        year = int(selected)
        return self.data[self.data["Year"] == year], selected

    # --------------------------------------------------------
    # HOME TAB
    # --------------------------------------------------------

    def build_home_tab(self):
        """Create the Home tab where user instructions are shown."""

        self.home_box = tk.Text(self.home_tab, font=("Arial", 12), wrap="word")
        self.home_box.pack(fill="both", expand=True, padx=15, pady=15)

    def update_home(self):
        """Show user guide and project explanation on the Home tab."""

        self.home_box.delete("1.0", tk.END)

        text = f"""
WORLD HAPPINESS REPORT ANALYSER

This software uses the World Happiness Report datasets from 2015 to 2019.
All five CSV files are loaded automatically when the application starts.

------------------------------------------------------------
STEP 1: SELECT YEAR
------------------------------------------------------------
Use the “Select Year” dropdown at the top of the window.

Available options:
• All Years
• 2015
• 2016
• 2017
• 2018
• 2019

The selected year controls:
• Analysis results
• Most charts
• Machine learning training
• Prediction model behaviour

If “All Years” is selected, the software uses the combined 2015–2019 dataset.

------------------------------------------------------------
STEP 2: TRAIN THE MODEL
------------------------------------------------------------
After selecting the year, click the “Train Model” button.

This trains the Linear Regression machine learning model using the selected data.

Important:
• Prediction will not work until the model is trained.
• If you change the selected year, train the model again.
• Training results appear in the “ML Results” tab.

------------------------------------------------------------
STEP 3: USE THE ANALYSIS TAB
------------------------------------------------------------
Go to the “Analysis” tab to view text-based results.

Available buttons:
• Show Summary — shows dataset size, countries, average score, and statistics.
• Top 10 Countries — shows the happiest countries for the selected year/data.
• Lowest 10 Countries — shows the lowest happiness countries for the selected year/data.

Each result clearly shows which year/data was used.

------------------------------------------------------------
STEP 4: USE THE CHARTS TAB
------------------------------------------------------------
Go to the “Charts” tab to create visual graphs.

Available charts:
• Average Happiness by Year
• Top 10 Happiest Countries
• Lowest 10 Countries
• GDP vs Happiness
• Correlation with Happiness

Important:
• “Average Happiness by Year” always uses all years automatically.
• Other charts use the year selected in the dropdown.
• Charts open in a separate matplotlib window.

------------------------------------------------------------
STEP 5: VIEW ML RESULTS
------------------------------------------------------------
Go to the “ML Results” tab after training the model.

This tab shows:
• Data used for training
• Training and testing rows
• MAE
• RMSE
• R² score
• Feature coefficients

Meaning:
• Lower MAE/RMSE means lower prediction error.
• R² closer to 1 means stronger model performance.

------------------------------------------------------------
STEP 6: USE THE PREDICT TAB
------------------------------------------------------------
Go to the “Predict” tab after training the model.

Enter values for:
• GDP
• Social Support
• Life Expectancy
• Freedom
• Generosity
• Corruption

Then click “Predict Happiness Score”.
The prediction uses the most recently trained model.

------------------------------------------------------------
STEP 7: USE THE DATA TABLE TAB
------------------------------------------------------------
Go to the “Data Table” tab to view the raw loaded dataset.

This helps users inspect the actual country records used by the software.

------------------------------------------------------------
WHAT NOT TO DO
------------------------------------------------------------
• Do not predict before training the model.
• Do not remove the CSV files from the application folder.
• Do not forget to train the model again after changing the selected year.
• Do not confuse “Average Happiness by Year” with selected-year charts; it always uses all years.

------------------------------------------------------------
CURRENT DATASET
------------------------------------------------------------
• Total rows loaded: {len(self.data)}
• Years available: 2015, 2016, 2017, 2018, 2019

Dataset factors used:
• GDP
• Social Support
• Life Expectancy
• Freedom
• Generosity
• Corruption

Important note:
The happiness factors already contribute to the happiness score.
Therefore, the prediction model is mainly useful for learning and demonstration.
"""
        self.home_box.insert(tk.END, text)

    # --------------------------------------------------------
    # ANALYSIS TAB
    # --------------------------------------------------------

    def build_analysis_tab(self):
        """Create buttons and output area for text-based analysis."""

        btn_frame = tk.Frame(self.analysis_tab)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Show Summary", command=self.show_summary, width=20).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Top 10 Countries", command=self.show_top_10, width=20).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Lowest 10 Countries", command=self.show_lowest_10, width=20).pack(side="left", padx=5)

        self.analysis_box = tk.Text(self.analysis_tab, font=("Courier", 11))
        self.analysis_box.pack(fill="both", expand=True, padx=15, pady=15)

    # --------------------------------------------------------
    # CHART TAB
    # --------------------------------------------------------

    def build_chart_tab(self):
        """Create chart selection dropdown and plot button."""

        control = tk.Frame(self.chart_tab)
        control.pack(pady=10)

        tk.Label(control, text="Select Chart:").pack(side="left", padx=5)

        self.chart_choice = tk.StringVar(value="Average Happiness by Year")

        chart_options = [
            "Average Happiness by Year",
            "Top 10 Happiest Countries",
            "Lowest 10 Countries",
            "GDP vs Happiness",
            "Correlation with Happiness"
        ]

        ttk.Combobox(
            control,
            textvariable=self.chart_choice,
            values=chart_options,
            state="readonly",
            width=32
        ).pack(side="left", padx=5)

        tk.Button(control, text="Plot Chart", command=self.plot_chart, width=16).pack(side="left", padx=5)

        tk.Label(
            self.chart_tab,
            text="Charts open in a separate matplotlib window.",
            font=("Arial", 12)
        ).pack(pady=30)

    # --------------------------------------------------------
    # MACHINE LEARNING TAB
    # --------------------------------------------------------

    def build_ml_tab(self):
        """Create the ML Results tab."""

        self.ml_box = tk.Text(self.ml_tab, font=("Courier", 11))
        self.ml_box.pack(fill="both", expand=True, padx=15, pady=15)
        self.ml_box.insert(tk.END, "Click 'Train Model' to train the model using the selected year.")

    # --------------------------------------------------------
    # PREDICTION TAB
    # --------------------------------------------------------

    def build_predict_tab(self):
        """Create input fields for prediction."""

        tk.Label(
            self.predict_tab,
            text="Predict Happiness Score",
            font=("Arial", 15, "bold")
        ).pack(pady=15)

        tk.Label(
            self.predict_tab,
            text="Prediction uses the latest trained model.",
            font=("Arial", 11)
        ).pack(pady=5)

        form = tk.Frame(self.predict_tab)
        form.pack()

        self.entries = {}

        # Example values are placed inside the boxes to guide the user.
        sample_values = {
            "GDP": "1.2",
            "Social Support": "1.3",
            "Life Expectancy": "0.8",
            "Freedom": "0.5",
            "Generosity": "0.2",
            "Corruption": "0.1"
        }

        for i, feature in enumerate(FEATURES):
            tk.Label(
                form,
                text=feature + ":",
                width=20,
                anchor="e"
            ).grid(row=i, column=0, pady=6, padx=5)

            entry = tk.Entry(form, width=15)
            entry.insert(0, sample_values[feature])
            entry.grid(row=i, column=1, pady=6, padx=5)

            self.entries[feature] = entry

        tk.Button(
            self.predict_tab,
            text="Predict Happiness Score",
            command=self.predict_happiness,
            width=25
        ).pack(pady=15)

        self.prediction_result = tk.StringVar(value="Train the model first, then predict.")

        tk.Label(
            self.predict_tab,
            textvariable=self.prediction_result,
            font=("Arial", 16, "bold")
        ).pack(pady=10)

    # --------------------------------------------------------
    # DATA TABLE TAB
    # --------------------------------------------------------

    def build_table_tab(self):
        """Create the tab where raw data is displayed."""

        self.table_frame = tk.Frame(self.table_tab)
        self.table_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def populate_table(self):
        """Display the loaded dataset in a table format."""

        # Clear the old table before creating a new one.
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        tree = ttk.Treeview(
            self.table_frame,
            columns=list(self.data.columns),
            show="headings"
        )

        # Create table headings.
        for col in self.data.columns:
            tree.heading(col, text=col)
            tree.column(col, width=130, anchor="center")

        # Insert all dataset rows into the table.
        for _, row in self.data.iterrows():
            tree.insert("", tk.END, values=list(row))

        # Add scrollbars so the user can move through the data.
        y_scroll = ttk.Scrollbar(self.table_frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(self.table_frame, orient="horizontal", command=tree.xview)

        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        tree.pack(side="left", fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")

    # --------------------------------------------------------
    # ANALYSIS FUNCTIONS
    # --------------------------------------------------------

    def show_summary(self):
        """Show basic summary statistics for the selected data."""

        selected_data, year_used = self.get_selected_data()

        self.analysis_box.delete("1.0", tk.END)

        self.analysis_box.insert(tk.END, "DATASET SUMMARY\n")
        self.analysis_box.insert(tk.END, "=" * 60 + "\n\n")
        self.analysis_box.insert(tk.END, f"Data used: {year_used}\n\n")

        self.analysis_box.insert(tk.END, f"Total rows: {len(selected_data)}\n")
        self.analysis_box.insert(tk.END, f"Total countries: {selected_data['Country'].nunique()}\n")
        self.analysis_box.insert(tk.END, f"Average Happiness Score: {selected_data['Score'].mean():.3f}\n\n")

        # If all years are selected, show average happiness score for each year.
        if year_used == "All Years":
            yearly = selected_data.groupby("Year")["Score"].mean()
            self.analysis_box.insert(tk.END, "Average Happiness Score by Year:\n")
            for year, score in yearly.items():
                self.analysis_box.insert(tk.END, f"{year}: {score:.3f}\n")

        self.analysis_box.insert(tk.END, "\nNumeric Summary:\n\n")
        self.analysis_box.insert(tk.END, str(selected_data.describe()))

    def show_top_10(self):
        """Show top 10 happiest countries for the selected data."""

        selected_data, year_used = self.get_selected_data()

        self.analysis_box.delete("1.0", tk.END)

        top = selected_data.sort_values(by="Score", ascending=False).head(10)

        self.analysis_box.insert(tk.END, "TOP 10 HAPPIEST COUNTRIES\n")
        self.analysis_box.insert(tk.END, "=" * 60 + "\n\n")
        self.analysis_box.insert(tk.END, f"Data used: {year_used}\n\n")

        for _, row in top.iterrows():
            self.analysis_box.insert(
                tk.END,
                f"{row['Country']} - Score: {row['Score']:.3f} - Year: {row['Year']}\n"
            )

    def show_lowest_10(self):
        """Show lowest 10 happiness countries for the selected data."""

        selected_data, year_used = self.get_selected_data()

        self.analysis_box.delete("1.0", tk.END)

        low = selected_data.sort_values(by="Score", ascending=True).head(10)

        self.analysis_box.insert(tk.END, "LOWEST 10 COUNTRIES\n")
        self.analysis_box.insert(tk.END, "=" * 60 + "\n\n")
        self.analysis_box.insert(tk.END, f"Data used: {year_used}\n\n")

        for _, row in low.iterrows():
            self.analysis_box.insert(
                tk.END,
                f"{row['Country']} - Score: {row['Score']:.3f} - Year: {row['Year']}\n"
            )

    # --------------------------------------------------------
    # CHART FUNCTION
    # --------------------------------------------------------

    def plot_chart(self):
        """Create the chart selected by the user."""

        choice = self.chart_choice.get()

        # This chart always uses all years because it compares yearly trends.
        if choice == "Average Happiness by Year":
            trend_data = self.data.groupby("Year")["Score"].mean()

            plt.figure(figsize=(8, 5))
            plt.plot(trend_data.index, trend_data.values, marker="o")
            plt.title("Average Happiness Score by Year\nData used: All Years")
            plt.xlabel("Year")
            plt.ylabel("Average Happiness Score")
            plt.grid(True)
            plt.show()
            return

        # Other charts use the selected year or all years based on the dropdown.
        selected_data, year_used = self.get_selected_data()

        if choice == "Top 10 Happiest Countries":
            top = selected_data.sort_values(by="Score", ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            plt.bar(top["Country"], top["Score"])
            plt.title(f"Top 10 Happiest Countries\nData used: {year_used}")
            plt.xlabel("Country")
            plt.ylabel("Happiness Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        elif choice == "Lowest 10 Countries":
            low = selected_data.sort_values(by="Score", ascending=True).head(10)

            plt.figure(figsize=(10, 6))
            plt.bar(low["Country"], low["Score"])
            plt.title(f"Lowest 10 Countries\nData used: {year_used}")
            plt.xlabel("Country")
            plt.ylabel("Happiness Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        elif choice == "GDP vs Happiness":
            plt.figure(figsize=(8, 5))
            plt.scatter(selected_data["GDP"], selected_data["Score"])
            plt.title(f"GDP vs Happiness Score\nData used: {year_used}")
            plt.xlabel("GDP")
            plt.ylabel("Happiness Score")
            plt.grid(True)
            plt.show()

        elif choice == "Correlation with Happiness":
            available_features = [f for f in FEATURES if f in selected_data.columns]
            correlations = selected_data[available_features + ["Score"]].corr(numeric_only=True)["Score"].drop("Score")

            plt.figure(figsize=(9, 5))
            plt.bar(correlations.index, correlations.values)
            plt.title(f"Correlation with Happiness Score\nData used: {year_used}")
            plt.xlabel("Factors")
            plt.ylabel("Correlation")
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.show()

    # --------------------------------------------------------
    # MACHINE LEARNING FUNCTION
    # --------------------------------------------------------

    def train_model(self):
        """Train the Linear Regression model using selected data."""

        selected_data, year_used = self.get_selected_data()

        available_features = [f for f in FEATURES if f in selected_data.columns]

        if len(selected_data) < 10:
            messagebox.showerror("Error", "Not enough data to train model.")
            return

        X = selected_data[available_features]
        y = selected_data["Score"]

        # Split data into training and testing sets.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Create and train the Linear Regression model.
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        self.model_features = available_features
        self.model_year_used = year_used

        # Test the model on unseen testing data.
        predictions = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        # Show machine learning results in the ML Results tab.
        self.ml_box.delete("1.0", tk.END)

        self.ml_box.insert(tk.END, "MACHINE LEARNING MODEL RESULTS\n")
        self.ml_box.insert(tk.END, "=" * 60 + "\n\n")
        self.ml_box.insert(tk.END, f"Data used for training: {year_used}\n\n")

        self.ml_box.insert(tk.END, "Model used: Linear Regression\n\n")
        self.ml_box.insert(tk.END, f"Training rows: {len(X_train)}\n")
        self.ml_box.insert(tk.END, f"Testing rows : {len(X_test)}\n\n")

        self.ml_box.insert(tk.END, f"MAE  : {mae:.4f}\n")
        self.ml_box.insert(tk.END, f"RMSE : {rmse:.4f}\n")
        self.ml_box.insert(tk.END, f"R²   : {r2:.4f}\n\n")

        self.ml_box.insert(tk.END, "Feature Coefficients:\n")
        self.ml_box.insert(tk.END, "-" * 60 + "\n")

        for feature, coef in zip(available_features, self.model.coef_):
            self.ml_box.insert(tk.END, f"{feature:<20}: {coef:+.4f}\n")

        self.ml_box.insert(tk.END, f"{'Intercept':<20}: {self.model.intercept_:+.4f}\n\n")

        self.ml_box.insert(tk.END, "Meaning:\n")
        self.ml_box.insert(tk.END, "- MAE and RMSE show prediction error. Lower is better.\n")
        self.ml_box.insert(tk.END, "- R² closer to 1 means stronger model performance.\n\n")

        self.status.set(f"Model trained using {year_used} | R²={r2:.4f}")

        # No graph is opened here. Graphs are handled only in the Charts tab.

    # --------------------------------------------------------
    # PREDICTION FUNCTION
    # --------------------------------------------------------

    def predict_happiness(self):
        """Predict happiness score from user-entered values."""

        if self.model is None:
            messagebox.showwarning("Model Not Trained", "Please train the model first.")
            return

        try:
            values = []

            # Read input values from the prediction boxes.
            for feature in self.model_features:
                value = float(self.entries[feature].get())
                values.append(value)

            # Convert user input into a dataframe for the model.
            input_data = pd.DataFrame([values], columns=self.model_features)
            prediction = self.model.predict(input_data)[0]

            self.prediction_result.set(
                f"Predicted Happiness Score: {prediction:.3f} | Model trained on: {self.model_year_used}"
            )

        except ValueError:
            messagebox.showerror("Input Error", "Please enter numbers only.")


# ------------------------------------------------------------
# START THE APPLICATION
# ------------------------------------------------------------

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = HappinessApp(root)
        root.mainloop()
    except Exception as error:
        messagebox.showerror("Application Error", str(error))