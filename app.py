# app.py - Main Gradio application for Hugging Face Spaces
import gradio as gr
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load your trained model
@gr.utils.cache_examples
def load_model():
    """Load the trained lung cancer prediction model"""
    try:
        with open('robust_lung_cancer_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model
model_data = load_model()

def predict_from_text(medical_text):
    """Predict cancer risk from medical text"""
    if not model_data or not medical_text.strip():
        return "Please enter medical text for analysis."
    
    try:
        # Load components
        text_model = model_data['text_model']
        text_vectorizer = model_data['text_vectorizer']
        text_processor = model_data['text_processor']
        
        # Clean and process text
        cleaned_text = text_processor.clean_medical_text_enhanced(medical_text)
        X_text = text_vectorizer.transform([cleaned_text])
        
        # Get prediction
        probability = text_model.predict_proba(X_text)[0, 1]
        prediction = text_model.predict(X_text)[0]
        
        # Risk interpretation
        if probability > 0.8:
            risk_level = "🔴 HIGH CANCER RISK"
            interpretation = "Strong textual indicators of malignancy"
            urgency = "Urgent medical evaluation recommended"
        elif probability > 0.6:
            risk_level = "🟡 MODERATE-HIGH RISK"
            interpretation = "Multiple concerning textual features"
            urgency = "Medical evaluation recommended within 2 weeks"
        elif probability > 0.4:
            risk_level = "🟡 MODERATE RISK"
            interpretation = "Some concerning features present"
            urgency = "Medical follow-up recommended"
        else:
            risk_level = "🟢 LOW RISK"
            interpretation = "Text suggests benign condition"
            urgency = "Routine follow-up appropriate"
        
        confidence = "High" if probability > 0.8 or probability < 0.2 else "Moderate"
        
        result = f"""
## 📊 Text Analysis Results

**Risk Assessment:** {risk_level}
**Probability Score:** {probability:.3f}
**Confidence Level:** {confidence}

**Clinical Interpretation:** {interpretation}
**Recommended Action:** {urgency}

**Model Version:** {model_data.get('model_version', 'Unknown')}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
*⚠️ This is an AI research tool. Always consult healthcare professionals for medical decisions.*
"""
        return result
        
    except Exception as e:
        return f"Error during text analysis: {str(e)}"

def predict_from_clinical_data(age, sex, smoking_pack_years, cea_level, hemoglobin, 
                              wbc_count, weight_loss, persistent_cough, dyspnea, chest_pain):
    """Predict cancer risk from clinical data"""
    if not model_data:
        return "Model not loaded. Please check model file."
    
    try:
        # Load components
        clinical_model = model_data['clinical_model']
        clinical_scaler = model_data['clinical_scaler']
        feature_extractor = model_data['feature_extractor']
        
        # Prepare patient data
        patient_data = {
            'age': age,
            'sex': sex,
            'smoking_pack_years': smoking_pack_years,
            'cea_level': cea_level,
            'hemoglobin': hemoglobin,
            'wbc_count': wbc_count,
            'weight_loss': 1 if weight_loss else 0,
            'persistent_cough': 1 if persistent_cough else 0,
            'dyspnea': 1 if dyspnea else 0,
            'chest_pain': 1 if chest_pain else 0
        }
        
        # Create features
        features = feature_extractor.create_clinical_features(patient_data)
        X_clinical = np.array([list(features.values())])
        X_clinical_scaled = clinical_scaler.transform(X_clinical)
        
        # Get prediction
        probability = clinical_model.predict_proba(X_clinical_scaled)[0, 1]
        
        # Enhanced risk stratification
        if probability > 0.85:
            risk_level = "🔴 VERY HIGH RISK"
            urgency = "IMMEDIATE"
            recommendation = "Urgent oncology referral and tissue biopsy within 48 hours"
            follow_up = "Immediate specialist consultation"
        elif probability > 0.7:
            risk_level = "🔴 HIGH RISK"
            urgency = "URGENT"
            recommendation = "Oncology referral and CT chest within 1-2 weeks"
            follow_up = "Specialist consultation within 2 weeks"
        elif probability > 0.5:
            risk_level = "🟡 MODERATE-HIGH RISK"
            urgency = "PRIORITY"
            recommendation = "CT chest and oncology consultation within 4 weeks"
            follow_up = "Reassessment in 4-6 weeks"
        elif probability > 0.3:
            risk_level = "🟡 MODERATE RISK"
            urgency = "ROUTINE+"
            recommendation = "Follow-up CT in 3 months, monitor symptoms closely"
            follow_up = "Reassessment in 3 months"
        else:
            risk_level = "🟢 LOW RISK"
            urgency = "ROUTINE"
            recommendation = "Standard screening schedule, annual low-dose CT if eligible"
            follow_up = "Annual screening if appropriate"
        
        # Patient summary
        age_group = 'Elderly (>75)' if age > 75 else 'Older adult (65-75)' if age > 65 else 'Adult (<65)'
        smoking_status = 'Heavy smoker (>30 py)' if smoking_pack_years > 30 else 'Moderate smoker (10-30 py)' if smoking_pack_years > 10 else 'Light/non-smoker (<10 py)'
        cea_status = 'Significantly elevated' if cea_level > 10 else 'Mildly elevated' if cea_level > 3 else 'Normal'
        symptom_count = sum([weight_loss, persistent_cough, dyspnea, chest_pain])
        symptom_burden = 'High' if symptom_count >= 3 else 'Moderate' if symptom_count >= 2 else 'Low'
        
        result = f"""
## 🏥 Clinical Risk Assessment

**Overall Risk:** {risk_level}
**Cancer Probability:** {probability:.3f}
**Urgency Level:** {urgency}

### 📋 Patient Profile
- **Age Group:** {age_group}
- **Smoking Status:** {smoking_status}
- **CEA Status:** {cea_status}
- **Symptom Burden:** {symptom_burden}

### 💡 Clinical Recommendations
**Primary Action:** {recommendation}
**Follow-up:** {follow_up}

### 🔍 Risk Factors Analysis
- **Age Risk:** {"⚠️ Elevated" if age > 65 else "✅ Standard"}
- **Smoking Risk:** {"🔴 High" if smoking_pack_years > 30 else "🟡 Moderate" if smoking_pack_years > 10 else "🟢 Low"}
- **Biomarker Risk:** {"🔴 High" if cea_level > 10 else "🟡 Moderate" if cea_level > 3 else "🟢 Normal"}
- **Symptom Risk:** {"🔴 High" if symptom_count >= 3 else "🟡 Moderate" if symptom_count >= 2 else "🟢 Low"}

**Model Version:** {model_data.get('model_version', 'Unknown')}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
*⚠️ This is an AI research tool for educational purposes. Always consult qualified healthcare professionals for medical decisions.*
"""
        return result
        
    except Exception as e:
        return f"Error during clinical analysis: {str(e)}"

# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Lung Cancer Risk Assessment") as demo:
        gr.Markdown("""
        # 🫁 Lung Cancer Risk Assessment Tool
        
        **AI-powered risk assessment using real medical literature data**
        
        This tool provides two types of analysis:
        1. **Text Analysis**: Analyze medical notes or clinical text
        2. **Clinical Data Analysis**: Assess risk based on patient demographics and lab values
        
        *⚠️ IMPORTANT: This is a research tool for educational purposes only. Always consult healthcare professionals for medical decisions.*
        """)
        
        with gr.Tabs():
            # Text Analysis Tab
            with gr.TabItem("📄 Text Analysis"):
                gr.Markdown("""
                ### Analyze Medical Text
                Enter clinical notes, medical reports, or any medical text to assess cancer risk indicators.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Medical Text",
                            placeholder="Enter medical notes, clinical findings, or patient presentation...",
                            lines=8,
                            max_lines=15
                        )
                        
                        text_examples = gr.Examples(
                            examples=[
                                ["Patient presents with persistent cough for 3 months, hemoptysis, and 15-pound weight loss. Chest CT shows 3cm spiculated mass in right upper lobe with mediastinal lymphadenopathy."],
                                ["Community-acquired pneumonia with fever, productive cough with purulent sputum. Chest X-ray shows bilateral lower lobe infiltrates. Patient responding well to antibiotic therapy."],
                                ["COPD exacerbation in 68-year-old male with 40 pack-year smoking history. Increased dyspnea and cough. Spirometry shows severe airflow obstruction. No acute distress on current bronchodilator therapy."]
                            ],
                            inputs=text_input
                        )
                        
                    with gr.Column(scale=2):
                        text_submit = gr.Button("🔍 Analyze Text", variant="primary", size="lg")
                        text_output = gr.Markdown(label="Analysis Results")
                
                text_submit.click(
                    fn=predict_from_text,
                    inputs=text_input,
                    outputs=text_output
                )
            
            # Clinical Data Analysis Tab
            with gr.TabItem("🏥 Clinical Data Analysis"):
                gr.Markdown("""
                ### Patient Clinical Assessment
                Enter patient demographics, laboratory values, and symptoms for risk assessment.
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 👤 Demographics")
                        age = gr.Slider(minimum=18, maximum=100, value=65, label="Age (years)")
                        sex = gr.Radio(choices=["M", "F"], value="M", label="Sex")
                        smoking_pack_years = gr.Slider(minimum=0, maximum=150, value=20, label="Smoking (pack-years)")
                        
                        gr.Markdown("#### 🧪 Laboratory Values")
                        cea_level = gr.Slider(minimum=0.1, maximum=100, value=2.5, label="CEA Level (ng/mL)")
                        hemoglobin = gr.Slider(minimum=6, maximum=18, value=13.5, label="Hemoglobin (g/dL)")
                        wbc_count = gr.Slider(minimum=2000, maximum=25000, value=7500, label="WBC Count (cells/μL)")
                    
                    with gr.Column():
                        gr.Markdown("#### 🩺 Symptoms")
                        weight_loss = gr.Checkbox(label="Unintentional Weight Loss")
                        persistent_cough = gr.Checkbox(label="Persistent Cough")
                        dyspnea = gr.Checkbox(label="Dyspnea (Shortness of Breath)")
                        chest_pain = gr.Checkbox(label="Chest Pain")
                        
                        clinical_submit = gr.Button("🔍 Assess Risk", variant="primary", size="lg")
                
                with gr.Row():
                    clinical_output = gr.Markdown(label="Risk Assessment Results")
                
                # Example patients
                gr.Markdown("#### 📋 Example Patients")
                with gr.Row():
                    high_risk_btn = gr.Button("High Risk Patient", variant="secondary")
                    low_risk_btn = gr.Button("Low Risk Patient", variant="secondary")
                    moderate_risk_btn = gr.Button("Moderate Risk Patient", variant="secondary")
                
                # Event handlers
                clinical_submit.click(
                    fn=predict_from_clinical_data,
                    inputs=[age, sex, smoking_pack_years, cea_level, hemoglobin, wbc_count,
                           weight_loss, persistent_cough, dyspnea, chest_pain],
                    outputs=clinical_output
                )
                
                # Example patient buttons
                high_risk_btn.click(
                    lambda: (70, "M", 45, 15.2, 9.1, 12500, True, True, True, False),
                    outputs=[age, sex, smoking_pack_years, cea_level, hemoglobin, wbc_count,
                            weight_loss, persistent_cough, dyspnea, chest_pain]
                )
                
                low_risk_btn.click(
                    lambda: (45, "F", 0, 1.8, 13.2, 7200, False, False, False, False),
                    outputs=[age, sex, smoking_pack_years, cea_level, hemoglobin, wbc_count,
                            weight_loss, persistent_cough, dyspnea, chest_pain]
                )
                
                moderate_risk_btn.click(
                    lambda: (58, "M", 25, 4.5, 11.8, 9500, False, True, False, True),
                    outputs=[age, sex, smoking_pack_years, cea_level, hemoglobin, wbc_count,
                            weight_loss, persistent_cough, dyspnea, chest_pain]
                )
        
        # About section
        with gr.Accordion("ℹ️ About This Tool", open=False):
            gr.Markdown("""
            ### Model Information
            - **Data Source**: Real PubMed medical literature (2,382 abstracts)
            - **Training**: Enhanced with overfitting prevention and validation
            - **Algorithms**: Logistic Regression with regularization
            - **Validation**: Cross-validated with robustness checks
            
            ### Features
            - **Text Analysis**: NLP analysis of medical text using TF-IDF vectorization
            - **Clinical Assessment**: Multi-factor risk analysis using demographics, labs, and symptoms
            - **Risk Stratification**: 5-level risk assessment with clinical recommendations
            
            ### Limitations
            - This is a research prototype, not a diagnostic tool
            - Results should be interpreted by qualified healthcare professionals
            - Not intended to replace clinical judgment or standard care protocols
            - Performance may vary on different populations or clinical settings
            
            ### Citation
            If you use this tool in research, please cite: *AI-Powered Lung Cancer Risk Assessment Tool - Research Prototype 2024*
            """)
    
    return demo

# Launch the application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()