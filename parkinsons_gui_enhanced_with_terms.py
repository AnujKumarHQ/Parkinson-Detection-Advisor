import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
import platform
import asyncio
from datetime import datetime

class ParkinsonsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parkinson's Disease Prediction Tool")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2b2b2b")
        
        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Arial", 10), padding=10, background="#3c78d8", foreground="white")
        style.map("TButton", background=[('active', '#5b9bd5')])
        style.configure("TLabel", font=("Arial", 10), background="#2b2b2b", foreground="white")
        style.configure("TEntry", font=("Arial", 10), fieldbackground="#3c3c3c", foreground="white")
        style.configure("TCombobox", font=("Arial", 10), fieldbackground="#3c3c3c", foreground="white")
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TNotebook", background="#2b2b2b")
        style.configure("TNotebook.Tab", font=("Arial", 10), padding=[10, 5], background="#3c3c3c", foreground="white")
        style.map("TNotebook.Tab", background=[('selected', '#3c78d8')])
        
        # Initialize data and model
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        self.data = None
        self.model = None
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.pretrained_model = None
        self.pretrained_scaler = None
        self.uploaded_data = None
        self.features = [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
            "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
            "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
            "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
        ]
        self.last_result = None
        
        # Setup GUI
        self.create_gui()
        
    def create_gui(self):
        container_frame = ttk.Frame(self.root, padding=15)
        container_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Left button panel
        button_frame = ttk.Frame(container_frame, padding=10)
        button_frame.grid(row=0, column=0, sticky="ns", padx=(0, 15))
        
        # Buttons
        self.train_button = ttk.Button(button_frame, text="Load & Train Model", command=self.train_model, width=18)
        self.train_button.grid(row=0, column=0, pady=6)
        self.predict_button = ttk.Button(button_frame, text="Predict on Test", command=self.predict_test, width=18)
        self.predict_button.grid(row=1, column=0, pady=6)
        self.test_button = ttk.Button(button_frame, text="Test", command=self.predict_single, width=18)
        self.test_button.grid(row=2, column=0, pady=6)
        self.upload_button = ttk.Button(button_frame, text="Upload Data", command=self.upload_data, width=18)
        self.upload_button.grid(row=3, column=0, pady=6)
        ttk.Button(button_frame, text="Export Results", command=self.export_results, width=18).grid(row=4, column=0, pady=6)
        ttk.Button(button_frame, text="Help", command=self.show_help, width=18).grid(row=5, column=0, pady=6)
        ttk.Button(button_frame, text="PDF Result", command=self.generate_pdf, width=18).grid(row=6, column=0, pady=6)
        ttk.Button(button_frame, text="About Us", command=self.show_about, width=18).grid(row=7, column=0, pady=6)
        ttk.Button(button_frame, text="Exit", command=self.exit_program, width=18).grid(row=8, column=0, pady=6)
        
        # Notebook for tabs
        notebook = ttk.Notebook(container_frame)
        notebook.grid(row=0, column=1, sticky="nsew")
        container_frame.columnconfigure(1, weight=1)
        container_frame.rowconfigure(0, weight=1)
        
        # Tabs
        prediction_frame = ttk.Frame(notebook, padding=10)
        treatment_frame = ttk.Frame(notebook, padding=10)
        notebook.add(prediction_frame, text="Prediction")
        notebook.add(treatment_frame, text="Treatment")
        
        # Tab content
        self.create_prediction_tab(prediction_frame)
        self.create_treatment_tab(treatment_frame)
    
    def exit_program(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to exit?"):
            self.root.destroy()
    
    def create_prediction_tab(self, main_frame):
        ttk.Label(main_frame, text="Parkinson's Disease Prediction", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        select_frame = ttk.LabelFrame(main_frame, text="Select Mode", padding=8)
        select_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=8, columnspan=2)
        
        ttk.Label(select_frame, text="Prediction Mode:").grid(row=0, column=0, padx=8, pady=5, sticky="w")
        self.mode_var = tk.StringVar(value="Manual Input")
        mode_combo = ttk.Combobox(select_frame, textvariable=self.mode_var, values=["Manual Input", "Pre-trained Model"], state="readonly", width=18)
        mode_combo.grid(row=0, column=1, padx=8, pady=5)
        mode_combo.bind("<<ComboboxSelected>>", self.toggle_mode)
        
        self.input_frame = ttk.LabelFrame(main_frame, text="Enter Patient Vocal Data", padding=8)
        self.input_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=8)
        
        self.entries = {}
        for i, feature in enumerate(self.features):
            ttk.Label(self.input_frame, text=feature).grid(row=i//2, column=(i%2)*2, padx=8, pady=4, sticky="w")
            entry = ttk.Entry(self.input_frame, width=12)
            entry.grid(row=i//2, column=(i%2)*2+1, padx=8, pady=4)
            self.entries[feature] = entry
        
        output_frame = ttk.LabelFrame(main_frame, text="Results", padding=8)
        output_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=8)
        
        self.result_text = scrolledtext.ScrolledText(output_frame, height=10, width=45, font=("Arial", 10), bg="#3c3c3c", fg="white")
        self.result_text.grid(row=0, column=0, padx=8, pady=8)
        
        terms_frame = ttk.LabelFrame(main_frame, text="Medical & Technical Terms", padding=8)
        terms_frame.grid(row=3, column=1, sticky="nsew", padx=8, pady=8)
        
        self.terms_text = scrolledtext.ScrolledText(terms_frame, height=12, width=45, font=("Arial", 10), bg="#3c3c3c", fg="white", wrap=tk.WORD)
        self.terms_text.grid(row=0, column=0, padx=8, pady=8)
        self.populate_terms()
        
        plot_frame = ttk.LabelFrame(main_frame, text="Visualizations", padding=8)
        plot_frame.grid(row=2, column=1, sticky="nsew", padx=8, pady=8)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, padx=8, pady=8)
        
        self.load_pretrained_model()
        self.toggle_mode()
    
    def create_treatment_tab(self, treatment_frame):
        ttk.Label(treatment_frame, text="Parkinson's Disease Management", font=("Arial", 16, "bold")).grid(row=0, column=0, pady=(0, 15), sticky="n")
        
        advice_frame = ttk.LabelFrame(treatment_frame, text="Recommendations", padding=8)
        advice_frame.grid(row=1, column=0, sticky="nsew", padx=97, pady=67)  # 10% margins (~97px, ~67px)
        treatment_frame.columnconfigure(0, weight=1)
        treatment_frame.rowconfigure(1, weight=1)
        
        advice_text = scrolledtext.ScrolledText(advice_frame, font=("Arial", 10), bg="#3c3c3c", fg="white", wrap=tk.WORD)
        advice_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        advice_frame.columnconfigure(0, weight=1)
        advice_frame.rowconfigure(0, weight=1)
        
        advice_content = """
Parkinson's Disease Management Recommendations
=============================================
These general recommendations are for individuals diagnosed with Parkinson's disease. Always consult a healthcare professional for personalized advice.

Lifestyle:
- Exercise Regularly: Engage in activities like walking, swimming, yoga, or tai chi for 30 minutes most days. Exercise improves mobility, balance, and mood.
- Sleep Well: Aim for 7-8 hours of quality sleep. Maintain a regular sleep schedule and create a restful environment.
- Manage Stress: Practice mindfulness, meditation, or deep breathing to reduce stress, which can worsen symptoms.
- Stay Socially Active: Join support groups or community activities to maintain emotional well-being and reduce isolation.

Diet:
- High-Fiber Foods: Eat fruits (e.g., berries, apples), vegetables (e.g., broccoli, spinach), and whole grains (e.g., oats, quinoa) to aid digestion and prevent constipation.
- Antioxidant-Rich Foods: Include berries, nuts, and green tea to combat oxidative stress.
- Healthy Fats: Consume omega-3-rich foods like salmon, walnuts, and flaxseeds for brain health.
- Hydration: Drink 8-10 glasses of water daily to stay hydrated and support overall health.
- Avoid Processed Foods: Limit sugary snacks, fried foods, and processed meats, which may increase inflammation.
- Protein Timing: If on levodopa, eat protein-rich foods (e.g., eggs, chicken) at times that don’t interfere with medication absorption (consult your doctor).

Medical Management:
- Medication Adherence: Take prescribed medications (e.g., levodopa, dopamine agonists) as directed to manage symptoms effectively.
- Regular Check-Ups: Visit a neurologist regularly to monitor disease progression and adjust treatments.
- Therapies: Consider physical therapy for mobility, occupational therapy for daily tasks, and speech therapy for voice issues.
- Deep Brain Stimulation: Discuss surgical options with your doctor if medications become less effective.

Support:
- Support Groups: Join local or online Parkinson’s support groups to share experiences and gain encouragement.
- Mental Health: Seek counseling or therapy to address depression or anxiety, common in Parkinson’s.
- Caregiver Resources: Involve family or caregivers and explore respite care options to support long-term care.

Disclaimer:
This advice is general and not a substitute for professional medical guidance. Consult your healthcare provider for a tailored treatment plan.
"""
        advice_text.insert(tk.END, advice_content)
        advice_text.config(state='disabled')
    
    def populate_terms(self):
        terms = """
Medical and Technical Terms:
• Parkinson's Disease: A neurological disorder affecting movement, causing tremors, stiffness, and slow movement (bradykinesia) due to loss of dopamine-producing brain cells.
• MDVP:Fo(Hz): Average vocal pitch frequency (Hertz). Lower or unstable Fo may indicate Parkinson's.
• MDVP:Fhi(Hz): Highest vocal pitch frequency. Variations reflect vocal instability.
• MDVP:Flo(Hz): Lowest vocal pitch frequency. Reduced range in Parkinson's.
• Jitter (%): Rapid vocal pitch variations. Higher in Parkinson's.
• Jitter:Abs: Absolute pitch difference between vocal cycles, in seconds.
• MDVP:RAP: Short-term pitch variation. Higher in Parkinson's.
• MDVP:PPQ: Longer-term pitch variation. Elevated in Parkinson's.
• Jitter:DDP: Composite jitter measure. Higher in Parkinson's.
• MDVP:Shimmer: Vocal loudness variations. Higher in Parkinson's.
• MDVP:Shimmer(dB): Shimmer in decibels.
• Shimmer:APQ3, APQ5, MDVP:APQ: Loudness variation measures. Higher in Parkinson's.
• Shimmer:DDA: Composite shimmer measure. Elevated in Parkinson's.
• NHR: Noise-to-harmonics ratio. Higher suggests breathy voice.
• HNR: Harmonics-to-noise ratio. Lower in Parkinson's.
• RPDE: Vocal signal complexity. Higher in Parkinson's.
• DFA: Long-term vocal signal correlations. Altered in Parkinson's.
• spread1, spread2: Nonlinear vocal variability. Higher in Parkinson's.
• D2: Dynamic vocal complexity. Increased in Parkinson's.
• PPE: Vocal pitch randomness. Higher in Parkinson's.
• Status: Target (1 = Parkinson's, 0 = Healthy).
• Random Forest Classifier: Machine learning model for prediction.
• Accuracy: Percentage of correct predictions.
• Classification Report: Precision, recall, F1-score summary.
• Confusion Matrix: Table of prediction outcomes.
• Cross-Validation: Model reliability assessment.
"""
        self.terms_text.insert(tk.END, terms)
        self.terms_text.config(state='disabled')
    
    def load_pretrained_model(self):
        try:
            self.data = pd.read_csv(self.url)
            self.data = self.data.drop('name', axis=1)
            X = self.data.drop('status', axis=1)
            y = self.data['status']
            X = X.fillna(X.mean())
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.pretrained_scaler = StandardScaler()
            X_train_scaled = self.pretrained_scaler.fit_transform(self.X_train)
            self.pretrained_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.pretrained_model.fit(X_train_scaled, self.y_train)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load pre-trained model: {str(e)}")
    
    def toggle_mode(self, event=None):
        mode = self.mode_var.get()
        if mode == "Pre-trained Model":
            for entry in self.entries.values():
                entry.config(state='disabled')
            self.train_button.config(state='disabled')
            self.test_button.config(state='disabled')
            self.upload_button.config(state='normal')
        else:
            for entry in self.entries.values():
                entry.config(state='normal')
            self.train_button.config(state='normal')
            self.test_button.config(state='normal')
            self.upload_button.config(state='disabled')
    
    def train_model(self):
        if self.mode_var.get() == "Pre-trained Model":
            messagebox.showinfo("Info", "Using pre-trained model. Manual training disabled.")
            return
        
        try:
            self.data = pd.read_csv(self.url)
            self.data = self.data.drop('name', axis=1)
            X = self.data.drop('status', axis=1)
            y = self.data['status']
            X = X.fillna(X.mean())
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train_scaled, self.y_train)
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Model trained successfully!\n")
            messagebox.showinfo("Success", "Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def upload_data(self):
        if self.mode_var.get() != "Pre-trained Model":
            messagebox.showinfo("Info", "Please switch to Pre-trained Model mode to upload data.")
            return
        
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if not file_path:
                return
            
            self.uploaded_data = pd.read_csv(file_path)
            if 'name' in self.uploaded_data.columns:
                self.uploaded_data = self.uploaded_data.drop('name', axis=1)
            
            if not all(col in self.uploaded_data.columns for col in self.features):
                messagebox.showerror("Error", "Uploaded CSV must contain all required features.")
                return
            
            X_uploaded = self.uploaded_data[self.features]
            X_uploaded = X_uploaded.fillna(X_uploaded.mean())
            self.X_test_scaled = self.pretrained_scaler.transform(X_uploaded)
            self.y_test = self.uploaded_data.get('status', None)
            self.predict_test()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload data: {str(e)}")
    
    def predict_test(self):
        model = self.pretrained_model if self.mode_var.get() == "Pre-trained Model" else self.model
        X_test_scaled = self.X_test_scaled
        y_test = self.y_test
        
        if model is None or X_test_scaled is None:
            messagebox.showerror("Error", "Please train the model or upload data!")
            return
        
        try:
            self.y_pred = model.predict(X_test_scaled)
            self.result_text.delete(1.0, tk.END)
            
            parkinsons_count = np.sum(self.y_pred == 1)
            healthy_count = np.sum(self.y_pred == 0)
            self.result_text.insert(tk.END, f"Predicted Parkinson's: {parkinsons_count} people\n")
            self.result_text.insert(tk.END, f"Predicted Healthy: {healthy_count} people\n\n")
            
            self.last_result = {
                'type': 'batch',
                'parkinsons_count': parkinsons_count,
                'healthy_count': healthy_count,
                'feature_importance': pd.Series(model.feature_importances_, index=self.features).nlargest(10),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if y_test is not None:
                accuracy = accuracy_score(y_test, self.y_pred)
                report = classification_report(y_test, self.y_pred)
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
                
                self.result_text.insert(tk.END, f"Accuracy: {accuracy:.2f}\n\n")
                self.result_text.insert(tk.END, "Classification Report:\n")
                self.result_text.insert(tk.END, report)
                self.result_text.insert(tk.END, f"\nCross-validation scores: {cv_scores}\n")
                self.result_text.insert(tk.END, f"Average CV score: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}\n")
                
                cm = confusion_matrix(y_test, self.y_pred)
                self.result_text.insert(tk.END, "\nConfusion Matrix:\n")
                self.result_text.insert(tk.END, f"{'':>10} {'Predicted 0':>12} {'Predicted 1':>12}\n")
                self.result_text.insert(tk.END, f"{'Actual 0':>10} {cm[0,0]:>12} {cm[0,1]:>12}\n")
                self.result_text.insert(tk.END, f"{'Actual 1':>10} {cm[1,0]:>12} {cm[1,1]:>12}\n")
                self.result_text.insert(tk.END, f"\nCorrect Predictions: {cm[0,0] + cm[1,1]} ({(cm[0,0] + cm[1,1])/cm.sum()*100:.1f}%)\n")
                self.result_text.insert(tk.END, f"Wrong Predictions: {cm[0,1] + cm[1,0]} ({(cm[0,1] + cm[1,0])/cm.sum()*100:.1f}%)\n")
                
                self.last_result.update({
                    'accuracy': accuracy,
                    'classification_report': report,
                    'cv_scores': cv_scores,
                    'confusion_matrix': cm
                })
                
                self.ax1.clear()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=self.ax1)
                self.ax1.set_title('Confusion Matrix')
                self.ax1.set_xlabel('Predicted')
                self.ax1.set_ylabel('Actual')
            else:
                self.ax1.clear()
                self.ax1.text(0.5, 0.5, "No ground truth for confusion matrix", ha='center', va='center')
                self.ax1.set_title('Confusion Matrix')
            
            self.ax2.clear()
            self.last_result['feature_importance'].plot(kind='barh', ax=self.ax2)
            self.ax2.set_title('Top 10 Feature Importance')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            if parkinsons_count > 0:
                self.result_text.insert(tk.END, "\nNote: Parkinson's detected. Visit 'Treatment' tab for advice.\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def predict_single(self):
        if self.mode_var.get() == "Pre-trained Model":
            messagebox.showinfo("Info", "Please switch to Manual Input mode to test a single patient.")
            return
        
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error", "Please train the model first!")
            return
        
        try:
            input_data = []
            for feature, entry in self.entries.items():
                value = entry.get()
                if not value:
                    messagebox.showerror("Error", f"Missing value for {feature}")
                    return
                input_data.append(float(value))
            
            input_data = np.array(input_data).reshape(1, -1)
            input_scaled = self.scaler.transform(input_data)
            prediction = self.model.predict(input_scaled)[0]
            
            result = "Parkinson's Detected" if prediction == 1 else "Healthy"
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Prediction for Patient: {result}\n\n")
            
            feature_importance = pd.Series(self.model.feature_importances_, index=self.features)
            top_features = feature_importance.nlargest(10)
            self.result_text.insert(tk.END, "Top 10 Feature Importances:\n")
            for feature, importance in top_features.items():
                self.result_text.insert(tk.END, f"{feature}: {importance:.4f}\n")
            
            self.last_result = {
                'type': 'single',
                'prediction': result,
                'input_data': pd.Series(input_data[0], index=self.features),
                'feature_importance': top_features,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.ax1.clear()
            self.ax1.text(0.5, 0.5, "No confusion matrix for single prediction", ha='center', va='center')
            self.ax1.set_title('Confusion Matrix')
            
            self.ax2.clear()
            top_features.plot(kind='barh', ax=self.ax2)
            self.ax2.set_title('Top 10 Feature Importance')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            if prediction == 1:
                self.result_text.insert(tk.END, "\nNote: Parkinson's detected. Visit 'Treatment' tab for advice.\n")
            
            messagebox.showinfo("Prediction", f"Prediction: {result}")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def generate_pdf(self):
        if self.last_result is None:
            messagebox.showerror("Error", "No results available. Please make a prediction first!")
            return
        
        try:
            latex_content = r"""
\documentclass[a4paper,10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{fancyhdr}
\geometry{margin=1in}
\pagestyle{fancy}
\fancyhf{}
\lhead{Parkinson's Disease Prediction Report}
\rhead{\today}
\cfoot{\thepage}

\begin{document}

\begin{center}
    \textbf{\Large Parkinson's Disease Prediction Report} \\
    \vspace{0.2cm}
    Generated on """ + self.last_result['timestamp'] + r"""
\end{center}

\section*{Prediction Results}
"""
            if self.last_result['type'] == 'single':
                latex_content += r"""
\textbf{Prediction}: """ + self.last_result['prediction'] + r""" \\
\vspace{0.2cm}
\textbf{Input Data}: \\
\begin{tabular}{lr}
\toprule
Feature & Value \\
\midrule
"""
                for feature, value in self.last_result['input_data'].items():
                    escaped_feature = feature.replace('_', '\\_').replace('%', '\\%')
                    latex_content += f"{escaped_feature} & {value:.4f} \\\\ \n"
                latex_content += r"""
\bottomrule
\end{tabular}
"""
                if self.last_result['prediction'] == "Parkinson's Detected":
                    latex_content += r"""
\section*{Management Recommendations}
\begin{itemize}
    \item \textbf{Exercise}: Walking, yoga, or tai chi for 30 min daily.
    \item \textbf{Diet}: High-fiber (berries, broccoli), omega-3s (salmon), stay hydrated.
    \item \textbf{Medication}: Follow prescriptions, consult neurologist.
    \item \textbf{Therapies}: Physical, occupational, speech therapy.
    \item \textbf{Support}: Join support groups, seek mental health resources.
\end{itemize}
\textbf{Disclaimer}: Consult a healthcare professional.
"""
            else:
                latex_content += r"""
\textbf{Predicted Parkinson's}: """ + str(self.last_result['parkinsons_count']) + r""" people \\
\textbf{Predicted Healthy}: """ + str(self.last_result['healthy_count']) + r""" people \\
"""
                if 'accuracy' in self.last_result:
                    latex_content += r"""
\textbf{Accuracy}: """ + f"{self.last_result['accuracy']*100:.2f}\\%" + r""" \\
\vspace{0.2cm}
\textbf{Classification Report}: \\
\begin{verbatim}
""" + self.last_result['classification_report'] + r"""
\end{verbatim}
\vspace{0.2cm}
\textbf{Cross-Validation Scores}: """ + f"{list(self.last_result['cv_scores'])}" + r""" \\
\textbf{Average CV Score}: """ + f"{self.last_result['cv_scores'].mean()*100:.2f}\\% ± {self.last_result['cv_scores'].std()*100:.2f}\\%" + r""" \\
\vspace{0.2cm}
\textbf{Confusion Matrix}: \\
\begin{tabular}{lrr}
\toprule
 & Predicted 0 & Predicted 1 \\
\midrule
Actual 0 & """ + str(self.last_result['confusion_matrix'][0,0]) + r""" & """ + str(self.last_result['confusion_matrix'][0,1]) + r""" \\
Actual 1 & """ + str(self.last_result['confusion_matrix'][1,0]) + r""" & """ + str(self.last_result['confusion_matrix'][1,1]) + r""" \\
\bottomrule
\end{tabular} \\
Correct Predictions: """ + str(self.last_result['confusion_matrix'][0,0] + self.last_result['confusion_matrix'][1,1]) + r""" \\
Wrong Predictions: """ + str(self.last_result['confusion_matrix'][0,1] + self.last_result['confusion_matrix'][1,0]) + r""" \\
"""
                if self.last_result['parkinsons_count'] > 0:
                    latex_content += r"""
\section*{Management Recommendations}
\begin{itemize}
    \item \textbf{Exercise}: Walking, yoga, or tai chi for 30 min daily.
    \item \textbf{Diet}: High-fiber (berries, broccoli), omega-3s (salmon), stay hydrated.
    \item \textbf{Medication}: Follow prescriptions, consult neurologist.
    \item \textbf{Therapies}: Physical, occupational, speech therapy.
    \item \textbf{Support}: Join support groups, seek mental health resources.
\end{itemize}
\textbf{Disclaimer}: Consult a healthcare professional.
"""
            
            latex_content += r"""
\section*{Feature Importance}
\begin{tabular}{lr}
\toprule
Feature & Importance \\
\midrule
"""
            for feature, importance in self.last_result['feature_importance'].items():
                escaped_feature = feature.replace('_', '\\_').replace('%', '\\%')
                latex_content += f"{escaped_feature} & {importance:.4f} \\\\ \n"
            latex_content += r"""
\bottomrule
\end{tabular}

\section*{Model Parameters}
\begin{itemize}
    \item Model: Random Forest Classifier
    \item Number of Estimators: 100
    \item Random State: 42
    \item Preprocessing: StandardScaler (zero mean, unit variance)
\end{itemize}

\end{document}
"""
            if platform.system() == "Emscripten":
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "PDF content (Pyodide):\n")
                self.result_text.insert(tk.END, latex_content)
                messagebox.showinfo("Success", "PDF content in results area (Pyodide mode).")
            else:
                file_path = filedialog.asksaveasfilename(defaultextension=".tex", filetypes=[("LaTeX files", "*.tex")])
                if not file_path:
                    return
                with open(file_path, 'w') as f:
                    f.write(latex_content)
                messagebox.showinfo("Success", f"LaTeX saved as {file_path}. Compile with latexmk -pdf.")
                
        except Exception as e:
            messagebox.showerror("Error", f"PDF generation failed: {str(e)}")
    
    def show_about(self):
        about_window = Toplevel(self.root)
        about_window.title("About Us")
        about_window.geometry("600x400")
        about_window.configure(bg="#2b2b2b")
        
        about_text = scrolledtext.ScrolledText(about_window, height=20, width=70, font=("Arial", 10), bg="#3c3c3c", fg="white", wrap=tk.WORD)
        about_text.pack(padx=10, pady=10)
        
        about_content = """
About Parkinson's Disease Prediction Tool
=======================================
Developed by: xAI-powered assistant (Grok)

Program Overview:
This tool predicts Parkinson's disease using vocal features from the UCI Parkinson's dataset. It provides a Tkinter GUI with manual input, batch predictions, and treatment advice. Pyodide-compatible for browser execution.

Dataset:
- Source: UCI Parkinson's Dataset (https://archive.ics.uci.edu/ml/datasets/parkinsons)
- Features: 22 vocal attributes (e.g., MDVP:Fo(Hz), Jitter, Shimmer, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE)
- Target: Status (1 = Parkinson's, 0 = Healthy)
- Preprocessing: Missing values filled with mean, scaled with StandardScaler

Model:
- Type: Random Forest Classifier
- Parameters: 100 estimators, random_state=42
- Training: 80% train, 20% test, scaled with StandardScaler
- Evaluation: Accuracy, classification report, confusion matrix, 5-fold cross-validation

Mechanisms:
- Manual Input: Train model, test single patient with feature importance.
- Pre-trained Model: Analyze uploaded CSV data.
- Visualizations: Confusion matrix, feature importance plots.
- PDF Report: LaTeX-based reports with predictions, metrics, treatment advice.
- Treatment Tab: Lifestyle, diet, medical advice for Parkinson’s.
- Export: Save results as text.
- Terms: Medical/technical term explanations.

Technology:
- GUI: Tkinter, dark theme (#2b2b2b, #3c78d8 buttons)
- Backend: Python, pandas, numpy, scikit-learn, matplotlib, seaborn
- PDF: LaTeX with PDFLaTeX (texlive-full)
- Compatibility: Pyodide, local file dialogs (non-Pyodide)

Powered by xAI:
This tool advances xAI's mission for AI-driven scientific discovery, offering accurate predictions for Parkinson’s.
"""
        about_text.insert(tk.END, about_content)
        about_text.config(state='disabled')
    
    def export_results(self):
        if self.y_pred is None:
            messagebox.showerror("Error", "No results to export. Please predict on test data!")
            return
        
        try:
            output = io.StringIO()
            parkinsons_count = np.sum(self.y_pred == 1)
            healthy_count = np.sum(self.y_pred == 0)
            output.write(f"Predicted Parkinson's: {parkinsons_count} people\n")
            output.write(f"Predicted Healthy: {healthy_count} people\n\n")
            
            if self.y_test is not None:
                output.write(f"Accuracy: {accuracy_score(self.y_test, self.y_pred):.2f}\n\n")
                output.write("Classification Report:\n")
                output.write(classification_report(self.y_test, self.y_pred))
                cv_scores = cross_val_score(self.model if self.mode_var.get() == "Manual Input" else self.pretrained_model, self.X_train_scaled, self.y_train, cv=5)
                output.write(f"\nCross-validation scores: {cv_scores}\n")
                output.write(f"Average CV score: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}\n")
                
                cm = confusion_matrix(self.y_test, self.y_pred)
                output.write("\nConfusion Matrix:\n")
                output.write(f"{'':>10} {'Predicted 0':>12} {'Predicted 1':>12}\n")
                output.write(f"{'Actual 0':>10} {cm[0,0]:>12} {cm[0,1]:>12}\n")
                output.write(f"{'Actual 1':>10} {cm[1,0]:>12} {cm[1,1]:>12}\n")
                output.write(f"\nCorrect Predictions: {cm[0,0] + cm[1,1]} ({(cm[0,0] + cm[1,1])/cm.sum()*100:.1f}%)\n")
                output.write(f"Wrong Predictions: {cm[0,1] + cm[1,0]} ({(cm[0,1] + cm[1,0])/cm.sum()*100:.1f}%)\n")
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Results exported to console (simulated in GUI).\n")
            self.result_text.insert(tk.END, output.getvalue())
            messagebox.showinfo("Success", "Results exported to console (in GUI).")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def show_help(self):
        help_text = """
Parkinson's Disease Prediction Tool
==================================
Uses UCI Parkinson's dataset to predict Parkinson's via vocal features.
- Dataset: 22 vocal features (e.g., MDVP:Fo(Hz), Jitter, Shimmer), status (1 = Parkinson's, 0 = Healthy).
- Modes:
  - Manual Input: Train model, test single patient.
  - Pre-trained Model: Analyze uploaded CSV.
- Features:
  - Train/use pre-trained model.
  - Predict on test/uploaded data (accuracy, classification report, confusion matrix, visualizations).
  - Test single patient (Manual Input) with feature importance.
  - Upload CSV (Pre-trained Model) for batch predictions.
  - View Parkinson’s/Healthy counts.
  - PDF reports with results, treatment advice.
  - Treatment tab: Lifestyle, medical advice.
  - Exit: Close the program.
  - Visualizations: Confusion matrix, feature importance.
- Terms: See 'Medical & Technical Terms' for feature/metric explanations.
"""
        messagebox.showinfo("Help", help_text)

async def main():
    root = tk.Tk()
    app = ParkinsonsGUI(root)
    root.mainloop()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        root = tk.Tk()
        app = ParkinsonsGUI(root)
        root.mainloop()